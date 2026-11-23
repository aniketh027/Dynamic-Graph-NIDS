import './register-tsx.mjs'
import assert from 'node:assert/strict'
import test from 'node:test'
import {readFileSync} from 'node:fs'
import React from 'react'
import {renderToStaticMarkup} from 'react-dom/server'
const {MemoryRouter}=await import('react-router')
const {LazyMotion,domAnimation}=await import('motion/react')
const {ReplayContext}=await import('../src/lib/replay-context.ts')
const {DemoContext}=await import('../src/lib/demo-context.ts')
const {BackendContext}=await import('../src/lib/api/context.ts')
const {ReplayResults,PredictionTable}=await import('../src/components/replay-results.tsx')
const {DetectionActivityPanel}=await import('../src/components/activity-panels.tsx')
const {Dashboard}=await import('../src/pages/dashboard.tsx')
const {IntrusionAlerts,NetworkMonitor,ModelPerformance}=await import('../src/pages/workspace-pages.tsx')
const {AlertCard}=await import('../src/components/alert-card.tsx')
const {AppShell}=await import('../src/components/app-shell.tsx')
const {eventTime}=await import('../src/lib/demo-data.ts')
const {EventsPanel}=await import('../src/components/events-panel.tsx')
const {TrafficFileInput}=await import('../src/components/traffic-file-input.tsx')
const {RealIntrusionDetails}=await import('../src/components/real-intrusion-drawer.tsx')
const wire=JSON.parse(readFileSync(new URL('./synthetic-wire-response.json',import.meta.url),'utf8').replace(/^\uFEFF/,''))
const result=structuredClone(wire.analysis)
result.predictions[0]={...result.predictions[0],prediction_label:'Bot',prediction_class_id:1,is_attack:true}
result.benign_prediction_count=1;result.intrusion_prediction_count=1;result.prediction_class_counts.Benign=1;result.prediction_class_counts.Bot=1
const noop=()=>{}
const { CSV_SCHEMA } = await import('../src/lib/csv-schema.generated.ts')
function render(component,state={phase:'complete',result,selected:null},demo=false,mock=false,events=[]){
  return renderToStaticMarkup(React.createElement(MemoryRouter,null,React.createElement(LazyMotion,{features:domAnimation},
    React.createElement(BackendContext.Provider,{value:{state:'offline',status:null,reason:'network'}},
      React.createElement(DemoContext.Provider,{value:{enabled:demo,events,mode:demo?'running':'idle',points:[],step:0,flowCount:0,reset:noop}},
        React.createElement(ReplayContext.Provider,{value:{state,session:{inspect:noop,select:noop},presentation:mock?'mock':undefined}},component))))))
}
test('real result summary uses returned counts, all classes, score label and no demo or performance metric',()=>{
  const html=render(React.createElement(ReplayResults,{result}))
  for(const text of ['REAL ANALYSIS','FROZEN MODEL','Model score','Trailing flows','SQL Injection','INTRUSION'])assert.ok(html.includes(text),text)
  assert.doesNotMatch(html,/SIMULATED|DEMO EVENT|confidence|probability|Critical|HIGH|Timestamp|Macro F1/)
})
test('table renders at most fifty rows and enables next page for 5000 predictions',()=>{
  const predictions=Array.from({length:5000},(_,i)=>({...result.predictions[0],window_index:i}))
  const html=render(React.createElement(PredictionTable,{predictions}))
  assert.equal((html.match(/<tr>/g)||[]).length,51)
  assert.match(html,/Page 1 of 100/);assert.match(html,/Inspect window 49/);assert.doesNotMatch(html,/Inspect window 50/)
})
test('dashboard and alerts consume real session results even if demo is also enabled',()=>{
  for(const Component of [Dashboard,IntrusionAlerts]) {
    const html=render(React.createElement(Component),{phase:'complete',result},true)
    assert.match(html,/Bot/);assert.match(html,/INTRUSION/);assert.doesNotMatch(html,/DEMO EVENT/)
  }
})
test('offline input is selectable while validation is disabled; no fabricated progress',()=>{
  const html=render(React.createElement(TrafficFileInput),{phase:'idle',file:null,validation:null,result:null,error:null})
  assert.match(html,/Browse files/);assert.match(html,/Backend offline/);assert.match(html,/disabled=""[^>]*>Validate file/)
  const active=render(React.createElement(TrafficFileInput),{phase:'analyzing',file:null,validation:null,result:null,error:null})
  assert.match(active,/RUNNING FROZEN HTGNN INFERENCE/);assert.doesNotMatch(active,/\d+%/)
})
test('Network Monitor stays disconnected and Model Performance does not consume replay counts',()=>{
  assert.match(render(React.createElement(NetworkMonitor)),/Live capture not connected/)
  const html=render(React.createElement(ModelPerformance));assert.doesNotMatch(html,/graphnids-c2b-synthetic|Total predictions/)
})
test('real drawer fields derive from predictions without fabricated severity, timestamp or network identity',()=>{
  const html=render(React.createElement(RealIntrusionDetails,{result,event:result.predictions[0]}))
  for(const text of ['Model score','Window index','Target flow index','61/78','70 nodes','69 edges','E2 HTGNN','seed 2026','Source filename','Analysis ID','File validated'])assert.ok(html.includes(text),text)
  assert.doesNotMatch(html,/\b(?:Timestamp|Critical|High|Medium|Low|Source IP|Destination IP|DEMO|SIMULATED|confidence|probability|packet captured|host isolated)\b/i)
})


test('mobile cards expose bounded prediction fields alongside the unchanged desktop table',()=>{
  const predictions=Array.from({length:105},(_,i)=>({...result.predictions[0],window_index:i,target_flow_index:i*5+4}))
  const html=render(React.createElement(PredictionTable,{predictions}))
  assert.equal((html.match(/class="replay-prediction-card"/g)||[]).length,50)
  for(const text of ['Prediction','Status','Window','Target flow','Model score','Inspect window 49','0.'])assert.ok(html.includes(text),text)
  assert.match(html,/Page 1 of 3/);assert.doesNotMatch(html,/Inspect window 50/)
})
test('zero intrusion predictions have no pagination; a single prediction still has normal controls',()=>{
  const empty=render(React.createElement(IntrusionAlerts),{phase:'complete',result:wire.analysis})
  assert.match(empty,/No intrusion predictions/)
  assert.doesNotMatch(empty,/<table|aria-label="Prediction table"|replay-pagination|Previous|Next|Page 1 of|Event severity legend/)
  const one=render(React.createElement(PredictionTable,{predictions:[result.predictions[0]]}))
  assert.match(one,/Page 1 of 1/);assert.match(one,/Previous/);assert.match(one,/Next/)
})
test('file row state is neutral before validation, amber during validation, green only after success',()=>{
  const base={file:new File(['x'],'sample.csv'),result:null,validation:null,error:null}
  for(const [phase,validation,error,tone,label] of [
    ['file_selected',null,null,'selected','Selected; schema not validated'],
    ['validating',null,null,'validating','Validation in progress'],
    ['validated',wire.validation,null,'validated','Backend validated'],
    ['validation_error',null,{code:'INVALID_SCHEMA',message:'Invalid header'},'error','Validation not successful'],
    ['backend_offline',null,{code:'BACKEND_OFFLINE',message:'Offline'},'error','Validation not successful'],
    ['complete',wire.validation,null,'validated','Backend validated'],
  ]){
    const html=render(React.createElement(TrafficFileInput),{...base,phase,validation,error})
    assert.match(html,new RegExp('selected-file-row file-'+tone));assert.ok(html.includes(label))
    if(!validation)assert.doesNotMatch(html,/file-validated|Backend validated/)
  }
})
test('severity legend appears only in demo, never idle or real replay',()=>{
  const idle=render(React.createElement(IntrusionAlerts),{phase:'idle',result:null})
  assert.doesNotMatch(idle,/Event severity legend|Filter event severity/)
  const demo=render(React.createElement(IntrusionAlerts),{phase:'idle',result:null},true)
  assert.match(demo,/Event severity legend/);assert.match(demo,/CRITICAL/);assert.match(demo,/DEMO DATA/)
})
test('explicit mock presentation replaces real authenticity on all replay result surfaces',()=>{
  for(const component of [React.createElement(ReplayResults,{result,dashboard:true}),React.createElement(IntrusionAlerts),React.createElement(Dashboard),React.createElement(RealIntrusionDetails,{result,event:result.predictions[0]})]){
    const html=render(component,{phase:'complete',result},false,true)
    assert.doesNotMatch(html,/REAL ANALYSIS|FROZEN MODEL|[Rr]eal intrusion/)
    assert.match(html,/Mock|MOCKED/)
  }
  const html=render(React.createElement(ReplayResults,{result}),{phase:'complete',result},false,true)
  assert.match(html,/MOCKED UI VALIDATION/);assert.match(html,/NO MODEL OUTPUT/)
})

test('exact source profiles require explicit local preparation and never expose metadata values',()=>{
  for (const sourceProfile of ['standard_80','extended_84']) {
    const html=render(React.createElement(TrafficFileInput),{phase:'preparation_required',file:new File(['PRIVATE-METADATA'],'source.csv'),sourceProfile,preparedFile:null,preparation:null,validation:null,error:null})
    assert.match(html,/Source format recognized/);assert.match(html,/Prepare locally/)
    assert.match(html,/Only the resulting 78-feature CSV will be uploaded/)
    for(const field of CSV_SCHEMA.profiles[sourceProfile].removed_metadata_fields)assert.ok(html.includes(field))
    assert.doesNotMatch(html,/PRIVATE-METADATA|Backend validated/)
    assert.match(html,/disabled=""[^>]*>Validate file/)
    assert.doesNotMatch(html,/disabled=""[^>]*>Prepare locally/)
  }
})

test('prepared status is amber and distinct from backend validation, including offline',()=>{
  const preparedFile=new File(['canonical'],'graphnids-prepared.csv')
  const preparation={record_count:12,window_count:2,trailing_record_count:2,removed_metadata_fields:['Timestamp','Label'],source_sha256:'a'.repeat(64),prepared_sha256:'b'.repeat(64)}
  const html=render(React.createElement(TrafficFileInput),{phase:'prepared',file:new File(['source'],'source.csv'),sourceProfile:'standard_80',preparedFile,preparation,validation:null,error:null})
  assert.match(html,/Prepared locally — backend validation required/)
  assert.match(html,/Validate prepared file/);assert.match(html,/disabled=""[^>]*>Validate prepared file/)
  assert.match(html,/Source SHA-256 · local only/);assert.match(html,/backend authenticates only the prepared upload/)
  assert.doesNotMatch(html,/file-validated|Backend validated|Analyze traffic/)
  const validated=render(React.createElement(TrafficFileInput),{phase:'validated',file:new File(['source'],'source.csv'),sourceProfile:'standard_80',preparedFile,preparation,validation:wire.validation,error:null})
  assert.match(validated,/Backend validated/);assert.match(validated,/Model inference has not run yet/)
})

test('local inspection and preparation show indeterminate local status and cancellation',()=>{
  for(const phase of ['inspecting','preparing']) {
    const html=render(React.createElement(TrafficFileInput),{phase,file:new File(['source'],'source.csv'),sourceProfile:'extended_84',preparedFile:null,preparation:null,validation:null,error:null})
    assert.match(html,/Cancel local processing/);assert.match(html,/Browser worker only/)
    assert.doesNotMatch(html,/RUNNING FROZEN HTGNN|Backend validated|\d+%/)
  }
})

test('the demo event log distinguishes zero latency from an unreported latency',()=>{
  const event={severity:'info',timestamp:0,component:'Demo',title:'Sample event',description:'Synthetic test'};
  const html=render(React.createElement(EventsPanel),undefined,true,false,[
    {...event,id:'zero',latency:0}, {...event,id:'missing'}
  ]);
  assert.match(html,/<td>0\.0 ms<\/td>/);
  assert.match(html,/<td>—<\/td>/);
});

test('the prediction table declares each heading as a column header',()=>{
  const html=render(React.createElement(PredictionTable,{predictions:result.predictions}));
  assert.equal((html.match(/<th scope="col">/g)||[]).length,6);
});

test('the skip link targets a focusable main element outside the normal tab order',()=>{
  const html=render(React.createElement(AppShell));
  assert.match(html,/<a href="#main-content" class="skip-link">Skip to content<\/a>/);
  assert.match(html,/<main id="main-content" tabindex="-1"/);
});

test('alert cards render the Unix epoch as a timestamp and omit unreported times',()=>{
  const props={severity:'info',title:'Demo event',description:'Synthetic fixture'};
  const epoch=render(React.createElement(AlertCard,{...props,timestamp:0}));
  assert.match(epoch,/<time dateTime="1970-01-01T00:00:00\.000Z">05:30:00 IST<\/time>/i);
  const undated=render(React.createElement(AlertCard,props));
  assert.doesNotMatch(undated,/<time/);
});

test('demo chart and event clocks preserve IST across midnight',()=>{
  assert.equal(eventTime(18 * 3600 + 29 * 60 + 59),'23:59:59');
  assert.equal(eventTime(18 * 3600 + 30 * 60),'00:00:00');
});

test('the demo detection chart shows an empty ring when there are no detection events',()=>{
  const event={id:'preparing',severity:'info',timestamp:0,source:'demo'};
  const html=render(React.createElement(DetectionActivityPanel),undefined,true,false,[event]);
  assert.match(html,/0 benign and 0 intrusion demo events/);
  assert.match(html,/stroke-dasharray="2 17"/);
  assert.doesNotMatch(html,/NaN|Infinity/);
});


test('demo timestamps outside the JavaScript date range render safely',()=>{
  const props={severity:'info',title:'Demo event',description:'Synthetic fixture'};
  const html=render(React.createElement(AlertCard,{...props,timestamp:Infinity}));
  assert.match(html,/<span>Time unavailable<\/span>/);
  assert.doesNotMatch(html,/<time|Time unavailable IST/);
  assert.equal(eventTime(NaN),'Time unavailable');
  assert.equal(eventTime(1e100),'Time unavailable');
});
