import json
import shutil
import subprocess
import unittest

from feedwatch.i18n import i18n_javascript, localize_html, translate_text


class LocalizationTest(unittest.TestCase):
    def test_html_translation_keeps_controls_and_is_idempotent(self):
        source='<label for="amount">Peixes detectados</label><input id="amount" name="amount" value="1,25"><code>Peixes detectados</code>'
        result=localize_html(source)
        self.assertIn('Fish detected',result)
        self.assertIn('name="amount" value="1,25"',result)
        self.assertIn('<code>Peixes detectados</code>',result)
        self.assertEqual(localize_html(result),result)

    def test_numeric_templates_accept_new_evidence_without_changing_values(self):
        source='Curva da GRU operacional: na etapa 22, acerto de 87,6% no treino e 61,3% na validação de outro dia.'
        english=translate_text(source)
        self.assertIn('epoch 22',english)
        self.assertIn('87.6%',english)
        self.assertIn('61.3%',english)

    def test_charts_have_language_variants_without_invalid_option_markup(self):
        result=localize_html('<title>Demonstração</title><select><option>Demonstração</option></select><img src="learning/class-recall.svg" alt="Acerto por categoria">')
        self.assertIn('class-recall.en.svg',result)
        self.assertIn('data-fw-graph-pt="learning/class-recall.svg"',result)
        self.assertNotIn('<option><span',result)
        self.assertNotIn('<title><span',result)

    @unittest.skipUnless(shutil.which('node'),'Node.js is required for browser-service checks')
    def test_language_preference_storage_fallback_and_numeric_formatting(self):
        script=i18n_javascript()
        harness=r'''
const vm=require('node:vm'),assert=require('node:assert/strict');
const script=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
function context(saved,blocked=false){
  const events={},storage=new Map(saved?[['feedwatch-language',saved]]:[]);
  const document={readyState:'complete',documentElement:{lang:''},querySelectorAll:()=>[],
    addEventListener:(name,fn)=>{events[name]=(events[name]||[]).concat(fn);},dispatchEvent:()=>{}};
  const window={addEventListener:()=>{}};
  const context={window,document,navigator:{language:'pt-BR'},CustomEvent:class{constructor(name,args){this.name=name;this.detail=args.detail;}},
    localStorage:{getItem:key=>{if(blocked)throw Error('blocked');return storage.get(key)||null;},setItem:(key,value)=>{if(blocked)throw Error('blocked');storage.set(key,value);}}};
  vm.createContext(context);vm.runInContext(script,context);
  return {context,api:window.FeedWatchI18n,document,storage,events};
}
const fresh=context();assert.equal(fresh.api.language,'en');assert.equal(fresh.document.documentElement.lang,'en');
assert.equal(fresh.api.number(1234.5,1,1),'1,234.5');
assert.equal(fresh.api.t('Curva da GRU operacional: na etapa 22, acerto de 87,6% no treino e 61,3% na validação de outro dia.'),
 'Operational GRU curve: at epoch 22, accuracy was 87.6% on training data and 61.3% on validation data from another day.');
let calls=0;const unsubscribe=fresh.api.subscribe(()=>calls++);
fresh.api.setLanguage('pt');assert.equal(calls,1);assert.equal(fresh.api.number(1234.5,1,1),'1.234,5');
assert.equal(fresh.storage.get('feedwatch-language'),'pt');assert.equal(fresh.document.documentElement.lang,'pt-BR');
unsubscribe();fresh.api.setLanguage('en');assert.equal(calls,1);
fresh.api.setLanguage('pt');vm.runInContext(script,fresh.context);assert.equal(fresh.api.language,'pt');assert.equal(fresh.events.change.length,1);
assert.equal(context('pt').api.language,'pt');assert.equal(context('invalid').api.language,'en');
const blocked=context(null,true);blocked.api.setLanguage('pt');assert.equal(blocked.api.language,'pt');
'''
        result=subprocess.run(['node','-e',harness],input=json.dumps(script),capture_output=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
