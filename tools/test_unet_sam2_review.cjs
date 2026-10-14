const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const assert=require('assert');
(async()=>{const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH,args:['--no-sandbox']});
try{
 const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/round-data/manifest.json',async route=>{
  const response=await route.fetch(),m=await response.json(),r=m.records[0];
  r.status='complete';r.scores=r.id+'/scores/frame_scores.json';r.csv=r.id+'/scores/frame_scores.csv';
  r.media=Object.fromEntries(['unet','forward','reverse'].map(s=>[s,{overlay:r.id+'/unet/overlay.mp4',segmentation:r.id+'/unet/mask_visual.mp4'}]));
  await route.fulfill({json:{...m,records:[r]}});
 });
 await page.route('**/scores/frame_scores.json',r=>r.fulfill({json:{policy:{training_selection:'NONE'},frames:Array.from({length:100},(_,f)=>({frame:f,time_seconds:f/30,prompt_frame:f===0,score:90,consistency_min:.9,consistency_mean:.9,transient_max:0,temporal_observable:f>2,directional_conflict:false,all_empty_classes:['object'],classes:{lefttool:{iou:{fr:.9,fu:.95,ru:.9},transient:{unet:0,forward:0,reverse:0},score:90,conflict_iou:false,conflict_components:false},object:{iou:{fr:null,fu:null,ru:null},transient:{unet:0,forward:0,reverse:0},score:null,conflict_iou:false,conflict_components:false}}}))}}));
 await page.goto('http://127.0.0.1:7861/round-review');await page.locator('#results').waitFor({state:'visible'});
 assert.equal(await page.locator('#gallery img').count(),10);assert((await page.locator('#scores').textContent()).includes('N/A'));
 await page.waitForFunction(()=>['unet','forward','reverse'].every(id=>document.getElementById(id).readyState>=1));
 await page.locator('#frame').fill('10');await page.locator('#go').click();assert((await page.locator('#frameInfo').textContent()).includes('第 10/'));
 await page.locator('#next').click();assert((await page.locator('#frameInfo').textContent()).includes('第 11/'));
 await page.locator('#mode').selectOption('segmentation');assert((await page.locator('#forward').getAttribute('src')).includes('mask_visual.mp4'));
 assert((await page.locator('#csv').getAttribute('href')).endsWith('frame_scores.csv'));
 assert.deepEqual(errors,[]);console.log('PASS round viewer: prompts, N/A scores, frame navigation, mode switch, CSV link; no data writes');
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exit(1)});
