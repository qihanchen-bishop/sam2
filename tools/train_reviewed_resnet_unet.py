#!/usr/bin/env python3
"""Snapshot reviewed masks, random-frame 80/20 split, train pretrained ResNet U-Net."""
import argparse
import hashlib
import json
import os
import random
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from train_light_semseg import (ImageSemSegDataset, prebuilt_image_samples, class_weights,
                               foreground_dice_loss, evaluate, save_previews, MEAN, STD)

LABELS = ['background', 'occluder', 'object', 'region', 'lefttool', 'righttool']
PALETTE = [[0,0,0], [64,160,255], [255,105,97], [119,221,119], [255,209,102], [101,214,173]]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.convs(torch.cat([x, skip], 1))


class ResNetUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool, self.layer1 = encoder.maxpool, encoder.layer1
        self.layer2, self.layer3, self.layer4 = encoder.layer2, encoder.layer3, encoder.layer4
        self.up3 = Decoder(512, 256, 128)
        self.up2 = Decoder(128, 128, 64)
        self.up1 = Decoder(64, 64, 32)
        self.up0 = Decoder(32, 64, 24)
        self.full = nn.Sequential(nn.Conv2d(24, 16, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, len(LABELS), 1))

    def forward(self, x):
        size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        z = self.layer4(s3)
        z = self.up0(self.up1(self.up2(self.up3(z, s3), s2), s1), s0)
        return self.full(F.interpolate(z, size=size, mode='bilinear', align_corners=False))


def prepare(args):
    if args.prepared_cache is not None:
        cache = args.prepared_cache.resolve()
        manifest = json.loads((cache / 'manifest.json').read_text())
        assert manifest['labels'] == LABELS
        assert manifest['seed'] == args.seed and manifest['image_size'] == [args.width, args.height]
        held_out = set(manifest.get('test_episodes', []))
        assert not any(s['episode'] in held_out for s in manifest['samples'])
        return cache
    cache = args.work_dir / 'cache'
    manifest_path = cache / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['seed'] == args.seed and manifest['image_size'] == [args.width, args.height]
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    def get(path):
        with urllib.request.urlopen(args.api + path, timeout=30) as response:
            return json.load(response)
    bootstrap = get('/api/bootstrap')
    sources, samples = [], []
    for ep in args.episodes:
        item = next(i for i in bootstrap['items'] if i['episode_index'] == ep and i['view'] == 'front')
        preview = get('/api/propagation?workspace_id=' + bootstrap['workspace_id'] + '&id=' + item['item_id'])['preview']
        review = preview['direction_review']
        assert not preview['stale'] and review['status'] == 'finalized' and review['unresolved'] == 0
        assert [o['name'] for o in preview['objects']] == LABELS[1:]
        root = Path(preview['output_path']) / review['final_subdir']
        sources.append({'episode': ep, 'item': item, 'run_id': preview['run_id'], 'final_root': str(root),
                        'review': review})
        samples += [(ep, f) for f in range(item['frame_count'])]
    write_json(args.work_dir / 'source_snapshot.json', sources)
    rng = random.Random(args.seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    train_count = int(len(shuffled) * .8)
    train = set(shuffled[:train_count])
    records = []
    counts = {'train': np.zeros(len(LABELS), np.int64), 'val': np.zeros(len(LABELS), np.int64)}
    for source in sources:
        item, ep = source['item'], source['episode']
        mask_root = Path(source['final_root']) / 'mask_frames'
        assert sorted(p.name for p in mask_root.glob('*.png')) == [f'{i:05d}.png' for i in range(item['frame_count'])]
        cap = cv2.VideoCapture(item['video_path'])
        for f in range(item['frame_count']):
            ok, rgb = cap.read()
            assert ok, (ep, f, 'unreadable RGB')
            path = mask_root / f'{f:05d}.png'
            mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert mask is not None and mask.shape == rgb.shape[:2] and mask.dtype == np.uint8 and mask.max() < len(LABELS)
            split = 'train' if (ep, f) in train else 'val'
            rel = Path(split) / item['video_key'] / f'file-{ep:03d}' / f'{f:05d}'
            image_path = cache / 'images' / rel.with_suffix('.jpg')
            mask_path = cache / 'masks' / rel.with_suffix('.png')
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            resized = cv2.resize(mask, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
            assert cv2.imwrite(str(image_path), cv2.resize(rgb, (args.width,args.height), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
            assert cv2.imwrite(str(mask_path), resized)
            counts[split] += np.bincount(resized.ravel(), minlength=len(LABELS))
            records.append({'episode': ep, 'frame': f, 'split': split,
                            'source_mask_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        assert not cap.read()[0], 'RGB has extra frames'
        cap.release()
        print(f'cached ep{ep:03d}: {item["frame_count"]} frames', flush=True)
    manifest = {'labels': LABELS, 'seed': args.seed, 'split_method': 'global_random_frame_80_20',
                'train': train_count, 'val': len(samples)-train_count, 'samples': records,
                'pixel_counts': {s: c.tolist() for s,c in counts.items()}, 'image_size': [args.width,args.height],
                'warning': 'Adjacent frames share train/val. Not a cross-video generalization benchmark.',
                'included_marked_frames': True}
    write_json(manifest_path, manifest)
    return cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--api', default='http://127.0.0.1:7861')
    p.add_argument('--prepared-cache', type=Path, help='Prepared train/val cache; bypass reviewed-seed data loading')
    p.add_argument('--episodes', type=int, nargs='+', default=[0,20,40,60])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--width', type=int, default=600)
    p.add_argument('--height', type=int, default=456)
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--min-epochs', type=int, default=25)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--min-delta', type=float, default=.0005)
    p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if (args.work_dir / 'latest.pt').exists() and not args.resume:
        raise RuntimeError('Existing training requires explicit --resume')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    write_json(args.work_dir / 'training_config.json', {**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               'architecture': 'resnet18_unet', 'pretrained': 'ResNet18_Weights.IMAGENET1K_V1',
               'normalization_mean': MEAN.tolist(), 'normalization_std': STD.tolist()})
    cache = prepare(args)
    samples = {s: prebuilt_image_samples(cache, s) for s in ('train','val')}
    if args.prepared_cache is not None:
        manifest = json.loads((cache / 'manifest.json').read_text())
        actual = {s:{(v.file_index,v.frame_index) for v in group} for s,group in samples.items()}
        expected = {s:{(v['episode'],v['frame']) for v in manifest['samples'] if v['split']==s} for s in samples}
        assert actual == expected, 'Cache files differ from the authorized split manifest'
        assert not actual['train'] & actual['val'], 'Train/validation overlap'
        assert not {ep for group in actual.values() for ep,_ in group} & set(manifest.get('test_episodes', [])), 'Test frames in cache'
        assert all(actual.values()), 'Empty training or validation set'
    datasets = {s: ImageSemSegDataset(v, (args.width,args.height), s=='train', 0.0) for s,v in samples.items()}
    loaders = {s: DataLoader(d, batch_size=args.batch_size, shuffle=s=='train', num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers>0) for s,d in datasets.items()}
    device = torch.device(args.device)
    model = ResNetUNet(pretrained=not args.resume).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3,
                                                        threshold=args.min_delta, threshold_mode='abs', min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    weights = class_weights(samples['train'], len(LABELS), len(samples['train']), 0).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    start_epoch, best, anchor, stale, history = 1, -1., -1., 0, []
    if args.resume:
        ck = torch.load(args.work_dir/'latest.pt', map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        start_epoch, best, anchor, stale, history = ck['epoch']+1, ck['best'], ck['anchor'], ck['stale'], ck['history']
    print(f'training {len(samples["train"])} / validation {len(samples["val"])} on {device}; pretrained ResNet18 U-Net', flush=True)
    converged = False
    for epoch in range(start_epoch, args.epochs+1):
        model.train(); begin=time.time(); loss_sum=0.; seen=0
        for step,(images,targets) in enumerate(loaders['train'],1):
            images,targets=images.to(device,non_blocking=True),targets.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logits=model(images)
                loss=criterion(logits,targets)+foreground_dice_loss(logits.float(),targets)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer);scaler.update()
            loss_sum+=loss.item()*len(images);seen+=len(images)
            if step==1 or step%100==0:print(f'epoch {epoch} step {step}/{len(loaders["train"])} loss={loss_sum/seen:.5f}',flush=True)
        metrics=evaluate(model,loaders['val'],device,len(LABELS))
        score=float(np.mean([metrics[f'iou_{i}'] for i in range(1,len(LABELS))]))
        improved=score>best
        best=max(best,score)
        if score>anchor+args.min_delta:anchor,stale=score,0
        else:stale+=1
        scheduler.step(score)
        row={**metrics,'foreground_miou':score,'epoch':epoch,'train_loss':loss_sum/seen,
             'lr':optimizer.param_groups[0]['lr'],'seconds':time.time()-begin,'stale':stale,'best':best}
        history.append(row)
        ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'epoch':epoch,'metrics':row,'labels':LABELS,'architecture':'resnet18_unet',
            'image_size':[args.width,args.height],'best':best,'anchor':anchor,'stale':stale,'history':history}
        for name in (['latest.pt','best.pt'] if improved else ['latest.pt']):
            torch.save(ck,args.work_dir/(name+'.tmp'));os.replace(args.work_dir/(name+'.tmp'),args.work_dir/name)
        write_json(args.work_dir/'history.json',history)
        print(json.dumps(row),flush=True)
        if epoch>=args.min_epochs and stale>=args.patience:converged=True;break
    best_ck=torch.load(args.work_dir/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(best_ck['model']);model.eval()
    metrics=evaluate(model,loaders['val'],device,len(LABELS))
    write_json(args.work_dir/'evaluation.json',{'best_epoch':best_ck['epoch'],'stopped_epoch':history[-1]['epoch'],
        'converged_by_early_stopping':converged,'metrics':metrics,'per_class_iou':{n:metrics[f'iou_{i}'] for i,n in enumerate(LABELS)},
        'warning':'Random-frame validation, not cross-video generalization.'})
    import train_light_semseg as shared
    shared.PREVIEW_PALETTE=np.array(PALETTE,np.uint8)
    save_previews(model,datasets['val'],device,args.work_dir/'previews',16)
    example=torch.zeros(1,3,args.height,args.width,device=device)
    traced=torch.jit.trace(model,example)
    with torch.inference_mode():torch.testing.assert_close(traced(example),model(example))
    traced.save(str(args.work_dir/'model.ts'))
    write_json(args.work_dir/'model_config.json',{'architecture':'resnet18_unet','torchscript':'model.ts',
        'labels':LABELS,'input_size_wh':[args.width,args.height],'normalization_mean':MEAN.tolist(),
        'normalization_std':STD.tolist(),'palette_rgb':PALETTE,'best_checkpoint':'best.pt'})
    write_json(args.work_dir/'complete.json',{'converged':converged,'best_epoch':best_ck['epoch'],'finished_at':time.time()})
    print('COMPLETE',flush=True)


if __name__ == '__main__':
    main()
