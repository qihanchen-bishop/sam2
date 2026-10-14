// Browser regression test with intercepted review actions: no user choices written.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs = require('fs');
const assert = require('assert');
(async () => {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_PATH, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    let state = {status: 'pending', revision: 0, default_direction: 'forward', choices: {},
      count_frame_details: {'1208': [{class:'occluder',counts:{forward:2,reverse:1},areas:{forward:[1000,26],reverse:[1001]}}]},
      rule_version: 'whole_iou085_count_auto50_v2', unresolved: 1, conflict_frame_count: 1, thresholds: {iou: .85, min_difference_pixels: 16},
      intervals: [{id: 0, start: 1208, end: 1208, classes: ['object'], worst_iou: .70, structural_c:true}]};
    let correctionJob=null;
    let unetFixture=null, unetSubmitted=false, unetFiltered=false;
    let globalPropagation={item_id:'ep000060__observation.images.front', kind:'propagation', state:'running'};
    await page.route('**/api/propagation?*', async route => {
      const response = await route.fetch();
      const data = await response.json();
      assert(data.preview, 'An existing sample preview is required');
      data.job = correctionJob;
      data.supports_batch_fill_holes = true;
      data.propagation_job = globalPropagation;
      data.supports_unet=true;data.unet_preview=unetFixture;
      data.unet_filter_version='area20_distance_then_limits_v6';
      data.preview.stale = false;
      data.preview.direction_review = state;
      data.preview.objects = [{id: 1, name: 'object'}];
      await route.fulfill({json: data});
    });
    let manualSaved=false, batchSubmitted=false, batchFillSubmitted=false;
    await page.route('**/api/unet-segment',async route=>{
      const request=route.request().postDataJSON();assert(request.item_id.includes('front'));
      unetSubmitted=true;unetFixture={run_id:'unet-fixture',frame_count:1478,fps:30,seconds:12,output_path:'/fixture/unet_preview'};
      if(request.filter_existing){unetFiltered=true;unetFixture.run_id='unet-filtered';unetFixture.component_filter={version:'area20_distance_then_limits_v6'};}
      await route.fulfill({json:{state:'running',run_id:'unet-fixture'}});
    });
    await page.route('**/api/component-preview', async route => {
      const request=route.request().postDataJSON();
      assert([1208,1209].includes(request.start));
      assert.equal(request.direction,'final');
      const deleted=request.operations.some(o=>o.kind==='delete');
      await route.fulfill({json:{width:1200,height:912,count:deleted?0:1,
        runs:deleted?[]:[[120100,100,1]],components:deleted?[]:[{id:1,area:100,box:[100,100,100,1],point:[100,100]}]}});
    });
    await page.route('**/api/correct-candidate', async route => {
      const request=route.request().postDataJSON();
      if(request.batch_fill_holes){
        assert.deepEqual(request.operations,[{kind:'fill_holes'}]);
        assert.equal(request.start,1200);assert.equal(request.end,1208);
        assert.equal(request.batch_keep_largest,false);
        batchFillSubmitted=true;state={...state,revision:state.revision+1};
        await route.fulfill({json:{state:'running'}});return;
      }
      assert.equal(request.direction,'final');
      assert.equal(request.class_id,1);
      if(request.batch_keep_largest){
        assert.equal(request.start,1200);assert.equal(request.end,1208);
        assert.deepEqual(request.operations,[{kind:'keep_largest'}]);
        assert.equal(request.remove_area,0);assert.equal(request.fill_area,0);
        assert.equal(request.review_revision,state.revision);
        batchSubmitted=true;state={...state,revision:state.revision+1};
        await route.fulfill({json:{state:'running'}});return;
      }
      assert.equal(request.edits.length,2);
      const first=request.edits.find(e=>e.start===1208),second=request.edits.find(e=>e.start===1209);
      assert.deepEqual(first.operations.map(o=>o.kind),['cut','delete','morphology','fill_holes','keep_largest']);
      assert.equal(first.operations[2].remove_area,20);assert.equal(first.operations[2].fill_area,50);
      assert.equal(first.operations[0].width,3);assert.equal(second.operations[0].kind,'paint');manualSaved=true;
      assert.equal(request.review_revision,state.revision);
      assert.equal(state.status,'finalized');
      state={...state,revision:state.revision+1,final_subdir:'reviewed_versions/edited',can_undo_final_correction:true,
        last_final_correction:{start:1208,removed_pixels:10,filled_pixels:25}};
      correctionJob={kind:'correction',state:'complete',item_id:request.item_id,run_id:request.run_id};
      await route.fulfill({json:{state:'running'}});
    });
    await page.route('**/api/undo-correction', async route => {
      state={...state,revision:state.revision+1,final_subdir:'reviewed_versions/browser-fixture',can_undo_final_correction:false,last_final_correction:null};
      await route.fulfill({json:state});
    });
    await page.route('**/api/review-choice', async route => {
      const request = route.request().postDataJSON();
      state = {...state, status: 'pending', revision: state.revision + 1,
        choices: {'0': request.direction}, unresolved: 0};
      await route.fulfill({json: state});
    });
    await page.route('**/api/review-finalize', async route => {
      assert.equal(state.unresolved, 0);
      state = {...state, status: 'finalized', revision: state.revision + 1,
        final_subdir: 'reviewed_versions/browser-fixture'};
      await route.fulfill({json: {state: 'running'}});
    });
    let scanRequests=0;
    await page.route('**/api/final-scan',async route=>{scanRequests++;await route.fulfill({json:{state:'failed'}});});
    await page.route('**/api/revision-frame',async route=>{
      const request=route.request().postDataJSON();assert.equal(request.frame,1208);
      state.manual_revision_frames=request.action==='remove'?[]:[{frame:1208,done:request.action==='done'}];
      await route.fulfill({json:{frames:state.manual_revision_frames}});
    });
    // Reuse a valid image/video for media widgets without requesting unselected final data.
    const sample = process.env.SAMPLE_PREVIEW;
    await page.route('**/api/difference-frame?*',r=>{
      const frame=Number(new URL(r.request().url()).searchParams.get('frame'));
      return r.fulfill({json:{frame,different_pixels:3,forward_only:1,reverse_only:1,class_mismatch:1,image:'data:image/png;base64,'+fs.readFileSync(sample+'/mask_frames/01208.png').toString('base64')}});
    });
    await page.route('**/api/preview-frame?*', r => r.fulfill({contentType: 'image/png',
      body: fs.readFileSync(sample + '/mask_frames/01208.png')}));
    await page.route('**/api/{preview-video,unet-video}?*', r => {
      const data=fs.readFileSync(sample+'/mask_visual.mp4'),range=r.request().headers()['range'];
      if(range){const match=/bytes=(\d+)-(\d*)/.exec(range),start=Number(match[1]),end=match[2]?Math.min(Number(match[2]),data.length-1):data.length-1;return r.fulfill({status:206,contentType:'video/mp4',headers:{'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${data.length}`},body:data.subarray(start,end+1)});}
      return r.fulfill({contentType:'video/mp4',headers:{'Accept-Ranges':'bytes'},body:data});
    });
    await page.goto('http://127.0.0.1:7861', {waitUntil: 'domcontentloaded'});
    await page.locator('#directionReview').waitFor({state: 'visible'});
    assert.deepEqual(await page.locator('#conflictFilter option').evaluateAll(nodes=>nodes.map(n=>n.value)),['C','other']);
    assert(await page.locator('#finalizeDirections').isDisabled());
    assert(await page.locator('#propagate').isDisabled(), 'Another video propagation must block new propagation');
    const annotationFrame = await page.locator('#frame').inputValue();
    await page.locator('#conflictFilter').selectOption('C');
    await page.waitForFunction(()=>document.querySelector('#conflictInterval').selectedOptions[0]?.textContent.includes('[分量数不同]'));
    assert((await page.locator('#conflictSummary').textContent()).includes('整体 IoU'));
    assert(await page.locator('#reanalyzeDirections').isHidden());
    await page.locator('#jumpConflict').click();
    await page.waitForFunction(() => document.querySelector('#previewFrame').value === '1208');
    assert((await page.locator('#componentCountDetails').textContent()).includes('occluder：正向 2（26）｜反向 1'));
    await page.locator('#markLowerFrame').click();
    await page.waitForFunction(()=>document.querySelector('#reviewMarkedFrames').options.length===1);
    assert.equal(await page.locator('#frame').inputValue(), annotationFrame);
    await page.waitForFunction(()=>document.querySelector('#differenceStatus').textContent.includes('第 1208 帧'));
    assert(await page.locator('#differenceCanvas').isVisible());
    assert(await page.locator('#differenceCanvas').evaluate(c=>c.closest('.final-edit-view')===document.querySelector('#finalDisplayCanvas').closest('.final-edit-view')));
    assert(Math.abs((await page.locator('#differenceCanvas').boundingBox()).width-(await page.locator('#forwardCanvas').boundingBox()).width)<1);
    assert.equal(await page.locator('#resultPreviewSource').inputValue(),'difference');
    assert(await page.locator('#forwardCanvas').isVisible());
    assert(await page.locator('#reverseCanvas').isVisible());
    assert((await page.locator('#forwardVideo').getAttribute('src')).includes('source=forward'));
    assert((await page.locator('#reverseVideo').getAttribute('src')).includes('source=reverse'));
    await page.locator('#componentEditor > summary').click();
    await page.waitForFunction(()=>document.querySelector('#componentLoad').disabled);
    await page.locator('#chooseForward').click();
    await page.waitForFunction(()=>!document.querySelector('#finalizeDirections').disabled);
    await page.locator('#unselectedConflictsOnly').check();
    await page.waitForFunction(()=>document.querySelector('#conflictInterval').options.length===0);
    assert(await page.locator('#chooseForward').isDisabled());
    assert(await page.locator('#jumpConflict').isDisabled());
    assert((await page.locator('#conflictFilterStatus').textContent()).includes('没有未选择'));
    await page.locator('#unselectedConflictsOnly').uncheck();
    await page.waitForFunction(()=>document.querySelector('#conflictInterval').options.length===1);
    assert((await page.locator('#conflictInterval').textContent()).includes('已选正向'));
    await page.locator('#finalizeDirections').click();
    await page.locator('#finalPreviewLink').waitFor({state:'visible'});
    const upperFrameBefore=await page.locator('#frame').inputValue();
    const candidateFrameBefore=await page.locator('#previewFrame').inputValue();
    await page.locator('#finalFrame').fill('1207');await page.locator('#finalFrame').press('Enter');
    await page.waitForFunction(()=>document.querySelector('#finalTime').textContent.includes('第 1207 帧'),null,{timeout:10000}).catch(async error=>{console.log(await page.evaluate(()=>({time:document.querySelector('#finalTime').textContent,input:document.querySelector('#finalFrame').value,current:document.querySelector('#finalPreviewVideo').currentTime,duration:document.querySelector('#finalPreviewVideo').duration,ready:document.querySelector('#finalPreviewVideo').readyState})));throw error;});
    await page.locator('#finalNext').click();
    await page.waitForFunction(()=>document.querySelector('#finalTime').textContent.includes('第 1208 帧'));
    await page.locator('#finalPrev').click();
    await page.waitForFunction(()=>document.querySelector('#finalTime').textContent.includes('第 1207 帧'));
    assert.equal(await page.locator('#frame').inputValue(),upperFrameBefore);
    assert.equal(await page.locator('#previewFrame').inputValue(),candidateFrameBefore);
    await page.locator('summary').filter({hasText:'区间统一处理'}).click();
    await page.locator('#batchStart').fill('1200');await page.locator('#batchEnd').fill('1208');
    page.once('dialog',async dialog=>{assert(dialog.message().includes('共 9 帧'));await dialog.accept();});
    await page.locator('#batchLargest').click();
    await page.waitForFunction(()=>document.querySelector('#batchStatus').textContent.includes('已提交'));
    assert(batchSubmitted);
    page.once('dialog',async dialog=>{assert(dialog.message().includes('填充封闭背景孔洞'));await dialog.accept();});
    await page.locator('#batchFillHoles').click();
    await page.waitForFunction(()=>document.querySelector('#batchStatus').textContent.includes('已提交'));
    assert(batchFillSubmitted);
    assert(Math.abs((await page.locator('#finalDisplayCanvas').boundingBox()).width-(await page.locator('#forwardCanvas').boundingBox()).width)<1);
    await page.waitForFunction(()=>document.querySelector('#finalMarkedFrames').options.length===1);
    assert.equal(await page.locator('#repairScope').count(),0);
    assert.equal(await page.locator('#componentSmall').count(),0);
    assert.equal(await page.locator('#editReverse').count(),0);
    await page.locator('#editMarkedFrame').click();
    await page.waitForFunction(()=>document.querySelector('#componentList').children.length===1);
    assert(Math.abs((await page.locator('#finalEditCanvas').boundingBox()).width-(await page.locator('#forwardCanvas').boundingBox()).width)<1);
    const editBounds=await page.locator('#finalEditCanvas').boundingBox();
    const controlsBounds=await page.locator('#componentEditor').boundingBox();
    assert(controlsBounds.x>=editBounds.x+editBounds.width,'Editor controls should be to the right of the canvas');
    assert((await page.locator('#componentContext').textContent()).includes('合并结果'));
    assert.equal(await page.locator('#componentCanvas').count(),0);
    await page.locator('#finalDisplayMode').selectOption('segmentation');
    assert(await page.locator('#finalOpacityControls').isHidden());
    assert.deepEqual(await page.locator('#finalEditCanvas').evaluate(c=>Array.from(c.getContext('2d').getImageData(0,0,1,1).data)),[0,0,0,255]);
    await page.locator('#finalDisplayMode').selectOption('overlay');
    await page.waitForFunction(()=>document.querySelector('#finalEditCanvas').getContext('2d').getImageData(0,0,1,1).data[0]>0);
    assert.deepEqual(await page.locator('#componentMode option').evaluateAll(nodes=>nodes.map(n=>n.value)),['select','cut','paint','erase']);
    assert.equal(scanRequests,0);
    for(const mode of ['paint','erase']){
      await page.locator('#componentMode').selectOption(mode);
      await page.locator('#componentWidth').fill('12');
      const c=page.locator('#finalEditCanvas');await c.scrollIntoViewIfNeeded();
      const b=await c.boundingBox();await page.mouse.click(b.x+b.width*.3,b.y+b.height*.3);
      await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('1 步'));
      await page.locator('#componentUndo').click();
      await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('0 步'));
    }
    await page.locator('#componentMode').selectOption('cut');
    await page.locator('#componentWidth').fill('3');
    const editCanvas=page.locator('#finalEditCanvas');
    await editCanvas.scrollIntoViewIfNeeded();
    const box=await editCanvas.boundingBox();
    await page.mouse.move(box.x+box.width*.1,box.y+box.height*.1);
    await page.mouse.down();
    await page.mouse.move(box.x+box.width*.2,box.y+box.height*.2,{steps:3});
    await page.mouse.up();
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('1 步'));
    await page.locator('#componentList button').click();
    await page.locator('#componentDelete').click();
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('2 步'));
    await page.locator('#componentUndo').click();
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('1 步'));
    await page.locator('#componentList button').click();
    await page.locator('#componentDelete').click();
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('2 步'));
    await page.locator('#repairRemove').fill('20');
    await page.locator('#repairFill').fill('50');
    await page.locator('#repairApply').click();
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('3 步'));
    await page.locator('#componentFillAll').click();
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('4 步'));
    await page.locator('#componentLargest').click();
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('5 步'));
    await page.locator('#finalDisplayMode').selectOption('segmentation');
    assert((await page.locator('#componentStatus').textContent()).includes('5 步'));
    await page.locator('#finalDisplayMode').selectOption('overlay');
    assert(!manualSaved, 'Area preview must not save');
    await page.locator('#editNextFrame').click();
    await page.waitForFunction(()=>document.querySelector('#componentContext').textContent.includes('1209'));
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('0 步') && !document.querySelector('#repairClass').disabled);
    await page.locator('#componentMode').selectOption('paint');
    await page.locator('#finalEditCanvas').scrollIntoViewIfNeeded();
    const nextBox=await page.locator('#finalEditCanvas').boundingBox();
    await page.mouse.click(nextBox.x+nextBox.width*.3,nextBox.y+nextBox.height*.3);
    await page.waitForFunction(()=>document.querySelector('#multiDraftList').options.length===2);
    await page.locator('#editPrevFrame').click();
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('5 步'));
    await page.locator('#editFrameNumber').fill('1209');await page.locator('#editFrameNumber').press('Enter');
    await page.waitForFunction(()=>document.querySelector('#componentContext').textContent.includes('1209') && document.querySelector('#componentStatus').textContent.includes('1 步'));
    await page.locator('#editFrameNumber').fill('1208');await page.locator('#editGoFrame').click();
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('5 步'));
    assert(await page.locator('#finalEditCanvas').isVisible());
    assert.equal(await page.locator('#repairClass').inputValue(),'1');
    await page.locator('#componentSave').click();
    await page.waitForFunction(()=>document.querySelector('#componentStatus').textContent.includes('全部草稿已保存'));
    assert.equal(await page.locator('#multiDraftList option').count(),0);
    assert(manualSaved);
    assert.equal(state.status,'finalized');assert.equal(state.choices['0'],'forward');
    assert.equal(await page.locator('#frame').inputValue(),annotationFrame);
    await page.locator('#repairUndo').click();
    await page.waitForFunction(()=>document.querySelector('#finalPreviewStatus').textContent.includes('browser-fixture'));
    assert.equal(await page.locator('#previewFrame').inputValue(),'1208');
    await page.locator('#finalPreviewLink').waitFor({state:'visible'});
    assert((await page.locator('#finalPreviewLink').getAttribute('href')).includes('source=final'));
    assert(await page.locator('#finalDisplayCanvas').isVisible());
    await page.locator('#finalDisplayMode').selectOption('segmentation');
    assert.deepEqual(await page.locator('#finalDisplayCanvas').evaluate(c=>Array.from(c.getContext('2d').getImageData(0,0,1,1).data)),[0,0,0,255]);
    await page.locator('#finalDisplayMode').selectOption('overlay');
    assert((await page.locator('#finalPreviewVideo').getAttribute('src')).includes('source=final'));
    const pairBox=await page.locator('.preview-pair').boundingBox();
    const finalBox=await page.locator('#finalPreviewPanel').boundingBox();
    assert(finalBox.y>=pairBox.y+pairBox.height);
    await page.locator('#segmentationMethod').selectOption('unet');
    assert(await page.locator('#sam2Workflow').isHidden());
    assert(await page.locator('#unetStart').isDisabled());
    globalPropagation=null;
    await page.evaluate(()=>refreshPropagation());
    await page.waitForFunction(()=>!document.querySelector('#unetStart').disabled);
    await page.locator('#unetStart').click();
    await page.locator('#unetVideo').waitFor({state:'visible'});
    assert(unetSubmitted);
    await page.waitForFunction(()=>document.querySelector('#unetVideo').readyState>=1);
    await page.locator('#unetFrame').fill('10');await page.locator('#unetGo').click();
    await page.waitForFunction(()=>document.querySelector('#unetTime').textContent.includes('第 10 /'));
    await page.locator('#unetNext').click();
    await page.waitForFunction(()=>document.querySelector('#unetTime').textContent.includes('第 11 /'));
    await page.locator('#unetDisplay').selectOption('segmentation');
    assert((await page.locator('#unetVideo').getAttribute('src')).includes('mode=segmentation'));
    await page.locator('#unetFilter').click();
    await page.waitForFunction(()=>document.querySelector('#unetStatus').textContent.includes('已应用连通分量过滤'));
    assert(unetFiltered);
    await page.locator('#segmentationMethod').selectOption('sam2');
    assert(await page.locator('#unetWorkflow').isHidden());
    assert(await page.locator('#sam2Workflow').isVisible());
    assert.deepEqual(errors, []);
    console.log('PASS: conflict jump, direction preview, choice, finalization, final preview; no user data modified');
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
