// Run: node --test test_chat.js
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function page(chatFetch,modelsFetch=async()=>({ok:true,json:async()=>({data:[{id:'auto',name:'Best'}]})})){
  const nodes=new Map();
  const node=()=>({value:'',checked:false,disabled:false,style:{},classList:{toggle(){}},textContent:'',innerHTML:'',children:[],appendChild(child){this.children.push(child)},replaceChildren(...children){this.children=children;this.value=''},focus(){}});
  const document={getElementById(id){if(!nodes.has(id))nodes.set(id,node());return nodes.get(id)},createElement:node,querySelectorAll(){return []}};
  const fetch=(url,options)=>url.endsWith('/v1/models')?modelsFetch(url,options):chatFetch(url,options);
  const context=vm.createContext({document,location:{origin:'http://localhost'},fetch,TextDecoder,Date,console,AbortController,setTimeout,clearTimeout});
  vm.runInContext(fs.readFileSync(__dirname+'/static/chat.html','utf8').split('<script>')[1].split('</script>')[0],context);
  document.getElementById('inp').value='Hello';
  document.getElementById('mdl').value='auto';
  document.getElementById('str').checked=true;
  return {context,document};
}
function stream(parts){return new ReadableStream({start(controller){for(const part of parts)controller.enqueue(new TextEncoder().encode(part));controller.close()}})}
const chunk=(delta,finish_reason=null)=>({id:'chatcmpl-test',object:'chat.completion.chunk',created:1,model:'auto',choices:[{index:0,delta,finish_reason}]});
const event=value=>'data: '+JSON.stringify(value)+'\n\n';
const settled=()=>new Promise(resolve=>setImmediate(resolve));

test('models load from the authenticated API with safe labels and refresh when the key changes',async()=>{
  const calls=[];
  const {document}=page(undefined,async(url,options)=>{
    calls.push({url,options});
    if(!options.headers.Authorization)return{ok:false,status:401};
    return{ok:true,json:async()=>({data:[{id:'new-model',name:'<img src=x onerror=alert(1)>'},{id:'auto',name:'Best'}]})};
  });
  await settled();
  assert.match(document.getElementById('mst').textContent,/Enter API key/);
  assert.equal(document.getElementById('mdl').children.length,0);
  document.getElementById('key').value='test-key';
  await document.getElementById('key').onchange();
  const select=document.getElementById('mdl');
  assert.equal(calls.at(-1).options.headers.Authorization,'Bearer test-key');
  assert.equal(calls.at(-1).options.cache,'no-store');
  assert.equal(select.value,'auto');
  assert.equal(select.children[0].value,'new-model');
  assert.equal(select.children[0].textContent,'<img src=x onerror=alert(1)>');
  assert.equal(select.children[0].innerHTML,'');
  assert.equal(select.disabled,false);
  const count=calls.length;
  document.getElementById('key').onblur();
  assert.equal(calls.length,count);
});
test('model refresh keeps only returned models and a still available selection',async()=>{
  let data=[{id:'auto',name:'Best'},{id:'old',label:'Old model'}];
  const {context,document}=page(undefined,async()=>({ok:true,json:async()=>({data})}));
  await settled();
  const select=document.getElementById('mdl');
  select.value='old';
  await context.loadModels();
  assert.equal(select.value,'old');
  data=[{id:'auto',name:'Best'},{id:'new',display_name:'New model'}];
  await context.loadModels();
  assert.deepEqual(select.children.map(option=>option.value),['auto','new']);
  assert.equal(select.value,'auto');
  data=[{id:'new'}];
  await context.loadModels();
  assert.equal(select.value,'new');
  assert.equal(select.children[0].textContent,'new');
});
test('failed or empty model refresh clears stale choices and prevents sending',async()=>{
  let result={ok:true,json:async()=>({data:[{id:'old'}]})};
  let sent=0;
  const {context,document}=page(()=>{sent++},async()=>result);
  await settled();
  result={ok:false,status:503};
  await context.loadModels();
  assert.equal(document.getElementById('mdl').children.length,0);
  assert.equal(document.getElementById('mdl').disabled,true);
  assert.match(document.getElementById('mst').textContent,/Models unavailable: HTTP 503/);
  await context.go();
  assert.equal(sent,0);
  result={ok:true,json:async()=>({data:[]})};
  await context.loadModels();
  assert.match(document.getElementById('mst').textContent,/Empty or invalid model list/);
});
test('a stale model request cannot replace models loaded for a new API key',async()=>{
  const pending=[];
  const {context,document}=page(undefined,()=>new Promise(resolve=>{pending.push(resolve)}));
  document.getElementById('key').value='new-key';
  const latest=context.loadModels();
  pending[1]({ok:true,json:async()=>({data:[{id:'new-model'}]})});
  await latest;
  pending[0]({ok:true,json:async()=>({data:[{id:'stale-model'}]})});
  await settled();
  assert.equal(document.getElementById('mdl').value,'new-model');
  assert.equal(document.getElementById('mrf').disabled,false);
});

test('SSE handles split UTF-8, CRLF, multiple data lines and final unterminated record',async()=>{
  const {context}=page();
  const encoded=new TextEncoder().encode('data: 繁體\r\ndata: 中文\r\n\r\ndata:[DONE]');
  const body=new ReadableStream({start(c){for(const byte of encoded)c.enqueue(Uint8Array.of(byte));c.close()}});
  const events=[];
  for await(const value of context.streamEvents(body))events.push(value);
  assert.deepEqual(events,['繁體\n中文','[DONE]']);
});
for(const [name,parts,message] of [
  ['malformed JSON',['data: {oops}\n\n'],'JSON'],
  ['upstream error',[event({error:{message:'quota unavailable'}})],'quota unavailable'],
  ['truncated stream',[event(chunk({role:'assistant',content:'partial'}))],'Incomplete stream'],
])test(name+' is visible and does not enter assistant history',async()=>{
  const {context,document}=page(async()=>({ok:true,body:stream(parts)}));
  await context.go();
  assert.match(document.getElementById('fv').innerHTML,/failed/);
  assert.match(document.getElementById('db').textContent,new RegExp(message));
  assert.equal(vm.runInContext('H.filter(m=>m.role==="assistant").length',context),0);
  assert.equal(document.getElementById('btn').disabled,false);
});
test('tool call deltas accumulate and response metadata is escaped',async()=>{
  const parts=[event(chunk({role:'assistant'})),event({...chunk({tool_calls:[{index:0,id:'call-1',type:'function',function:{name:'calculator',arguments:'{"x":'}}]}),model:'<img src=x onerror=alert(1)>'}),event(chunk({tool_calls:[{index:0,function:{arguments:'1}'}}]})),event(chunk({},'tool_calls')),'data: [DONE]\n\n'];
  const {context,document}=page(async()=>({ok:true,body:stream(parts)}));
  await context.go();
  assert.equal(vm.runInContext('H[1].tool_calls[0].function.arguments',context),'{"x":1}');
  context.addM('a','safe',{fr:'stop',md:'<img src=x onerror=alert(1)>'});
  assert.match(document.getElementById('msgs').children.at(-1).innerHTML,/&lt;img/);
});
test('Enter and Clear cannot race an active request',async()=>{
  let resolve,calls=0;
  const {context,document}=page(()=>{calls++;return new Promise(r=>{resolve=r})});
  const running=context.go();
  document.getElementById('inp').value='second';
  await context.go();context.clr();
  assert.equal(calls,1);
  assert.equal(vm.runInContext('H.length',context),1);
  resolve({ok:false,status:503,text:async()=>'offline'});
  await running;
});
test('format validator rejects broken token arithmetic and accepts a trailing usage chunk',()=>{
  const {context}=page();
  const invalid={id:'chatcmpl-test',object:'chat.completion',created:1,model:'auto',choices:[{index:0,finish_reason:'stop',message:{role:'assistant',content:'four'}}],usage:{prompt_tokens:2,completion_tokens:3,total_tokens:99}};
  assert.equal(context.validateOpenAI(invalid,false).find(check=>check.name==='usage arithmetic').ok,false);
  const chunks=[chunk({role:'assistant',content:'four'}),chunk({},'stop'),{...chunk({}),choices:[],usage:{prompt_tokens:2,completion_tokens:3,total_tokens:5}}];
  chunks._gotDone=true;
  assert.equal(context.validateOpenAI(chunks,true).filter(check=>check.required&&!check.ok).length,0);
});
