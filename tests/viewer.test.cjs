const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const html = fs.readFileSync('observatorio/web/index.html', 'utf8');
const source = html.slice(html.indexOf('<script>') + 8, html.lastIndexOf('</script>'))
  .replace(/boot\(\)\.catch\([\s\S]*$/, '');
function setup(fetch, extra = {}) {
  const retry = {};
  const box = {dataset: {}, querySelector: () => retry, querySelectorAll: () => []};
  const ctx = vm.createContext({document: {querySelector: () => box}, fetch, ...extra});
  vm.runInContext(source, ctx);
  return {ctx, box, retry};
}
test('null statistics produce an unavailable legend', () => {
  const {ctx, box} = setup();
  vm.runInContext(`st.base=[{id:'x',path:[],stats:{turns_mean:null}}];
    st.overall={stats:{turns_mean:null}}; computeDomains();
    st.colorBy='turns_mean'; renderLegend();`, ctx);
  assert.match(box.innerHTML, /No data available/);
});
test('failed detail requests are evicted and can be retried', async () => {
  let calls = 0;
  const {ctx} = setup(async () => ++calls === 1
    ? {ok:false, status:503}
    : {ok:true, json:async () => ({id:'a'})});
  await assert.rejects(vm.runInContext("getDetail('a')", ctx), /503/);
  assert.equal((await vm.runInContext("getDetail('a')", ctx)).id, 'a');
  await vm.runInContext("getDetail('a')", ctx);
  assert.equal(calls, 2);
});
test('detail failure renders a retry instead of rejecting rendering', async () => {
  const {ctx, box, retry} = setup(async () => {throw Error('offline');});
  await vm.runInContext("st.selected=['a']; renderCards()", ctx);
  assert.match(box.innerHTML, /Could not load cluster details/);
  assert.equal(typeof retry.onclick, 'function');
});
test('older failed request cannot overwrite a newer cleared selection', async () => {
  let fail;
  const {ctx, box} = setup(() => new Promise((resolve, reject) => {fail=reject;}));
  const pending = vm.runInContext("st.selected=['a']; renderCards()", ctx);
  vm.runInContext("cardsToken++;", ctx);
  box.innerHTML = 'new selection';
  fail(Error('offline'));
  await pending;
  assert.equal(box.innerHTML, 'new selection');
});
test('failed search records a retryable error and a retry clears it', async () => {
  let calls = 0;
  const {ctx} = setup(async () => ++calls === 1
    ? {ok:false, status:500}
    : {ok:true, json:async () => ({q:'code', results:[], total:0})});
  vm.runInContext('renderAll = () => {};', ctx);
  await vm.runInContext("runSearch('code')", ctx);
  assert.match(vm.runInContext('st.search.error', ctx), /500/);
  await vm.runInContext("runSearch('code')", ctx);
  assert.equal(vm.runInContext('st.search.error', ctx), undefined);
});

test('static mode reads files and searches in the browser like /api/search', async () => {
  const urls = [];
  const index = [
    {id: 'L0-1', level: 0, name: 'Debug Python scripts', n: 50, description: 'Users asked.'},
    {id: 'L1-1', level: 1, name: 'Write code', n: 90, description: 'Users asked about Python.'},
    {id: 'L1-2', level: 1, name: 'Python tutoring', n: 10, description: 'Users asked.'},
    {id: 'L0-2', level: 0, name: 'Plan trips', n: 40, description: 'Users asked.'},
  ];
  const {ctx} = setup(async url => { urls.push(url); return {ok: true, json: async () => index}; },
                      {OBS_STATIC: true});
  assert.equal(vm.runInContext("URLS.cluster('L2-0')", ctx), 'data/cluster/L2-0.json');
  const r = await vm.runInContext("searchClusters('python')", ctx);
  // name matches first, then higher level, then larger n; description matches last
  const plain = JSON.parse(JSON.stringify(r));  // arrays from the vm realm have a different prototype
  assert.deepEqual(plain.results.map(h => h.id), ['L1-2', 'L0-1', 'L1-1']);
  assert.deepEqual(plain.results.map(h => h.match), ['name', 'name', 'description']);
  assert.equal(r.total, 3);
  await vm.runInContext("searchClusters('plan')", ctx);
  assert.deepEqual(urls, ['data/search.json']);  // index fetched once, then reused
});
