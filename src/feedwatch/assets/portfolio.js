function calculateFeedingScenario(usualKg, proposedKg, unitPrice=null) {
  if (![usualKg,proposedKg].every(n=>typeof n==='number' && Number.isFinite(n) && n>=0) ||
      (unitPrice!==null && (typeof unitPrice!=='number' || !Number.isFinite(unitPrice) || unitPrice<0))) {
    throw new Error('Informe quantidades e preço válidos, iguais ou maiores que zero.');
  }
  const result={usual_kg:usualKg,proposed_kg:proposedKg,difference_kg:usualKg-proposedKg,
    difference_percent:usualKg>0?(usualKg-proposedKg)/usualKg*100:null,
    usual_cost:unitPrice===null?null:usualKg*unitPrice,
    proposed_cost:unitPrice===null?null:proposedKg*unitPrice,
    difference_cost:unitPrice===null?null:(usualKg-proposedKg)*unitPrice};
  if(Object.values(result).some(n=>n!==null && !Number.isFinite(n)))throw new Error('Os valores são grandes demais para calcular.');
  return result;
}

function motionFrameAt(frames, time) {
  let low=0, high=frames.length-1, found=-1;
  while(low<=high) {
    const middle=(low+high)>>1;
    if(frames[middle].timestamp_seconds<=time+1e-9){found=middle;low=middle+1;}
    else high=middle-1;
  }
  return found<0?null:frames[found];
}

function motionChartData(frames, time, duration) {
  const visible=frames.filter(row=>row.timestamp_seconds<=time+1e-9);
  const valid=row=>row.status==='ok' && Number.isFinite(row.mean_source_px_per_second) && row.mean_source_px_per_second>=0;
  const maximum=Math.max(400,Math.ceil(Math.max(0,...visible.filter(valid).map(row=>row.mean_source_px_per_second))*1.1/20)*20);
  const step=Math.max(1,Math.ceil(visible.length/150));
  const length=Math.max(duration,0.001);
  let path='', move=true;
  for(const [index,row] of visible.entries()) {
    if(!valid(row)){move=true;continue;}
    if(index%step && index!==visible.length-1)continue;
    const x=300*Math.min(1,Math.max(0,row.timestamp_seconds/length));
    const y=66-62*row.mean_source_px_per_second/maximum;
    path+=(move?'M':'L')+x.toFixed(2)+','+y.toFixed(2);move=false;
  }
  return {path,maximum,cursor:300*Math.min(1,Math.max(0,time/length))};
}

async function resolveFeedWatchMedia(source, options={}) {
  const signal=options.signal, urls=options.urlApi||URL;
  const checkAbort=()=>{if(signal?.aborted)throw new DOMException('Media request aborted','AbortError');};
  const handle=(url,owned=false)=>{
    let released=false;
    return {url,release(){if(!released && owned)urls.revokeObjectURL(url);released=true;}};
  };
  const asBlob=blob=>{
    checkAbort();
    if(!blob.size)throw new Error('Empty media response');
    const result=handle(urls.createObjectURL(blob),true);
    if(signal?.aborted){result.release();checkAbort();}
    return result;
  };
  checkAbort();
  if(typeof Blob!=='undefined' && source instanceof Blob)return asBlob(source);
  if(typeof source!=='string' || !source)throw new TypeError('Invalid media source');
  if(/^(blob:|data:|file:)/i.test(source))return handle(source);
  const base=options.baseUrl||(typeof document!=='undefined'?document.baseURI:'http://localhost/');
  const crossOrigin=new URL(source,base).origin!==new URL(base).origin;
  let response;
  try {
    response=await (options.fetchImpl||fetch)(source,{headers:{Range:'bytes=0-0'},signal});
  } catch(error) {
    checkAbort();
    if(crossOrigin && error.name==='TypeError')return handle(source);
    throw error;
  }
  checkAbort();
  const cancel=async()=>{try{await response.body?.cancel();}catch{}checkAbort();};
  if(response.status===206) {
    const range=response.headers.get('Content-Range');
    await cancel();
    if(/^bytes\s+0-0\/(?:[1-9]\d*|\*)$/i.test(range||''))return handle(source);
    if(crossOrigin && !range)return handle(source);
    throw new Error('Invalid media range response');
  }
  if(crossOrigin && response.type==='opaque'){await cancel();return handle(source);}
  if(response.status!==200){await cancel();throw new Error('Media request failed ('+response.status+')');}
  const type=response.headers.get('Content-Type')||'';
  if(/^(?:text\/|application\/(?:json|problem\+json))/i.test(type)) {
    await cancel();throw new Error('Media response is not a video');
  }
  // A host ignoring Range already sent the full resource: consume it once.
  return asBlob(await response.blob());
}

function mountFeedWatch(root, data) {
  const I=window.FeedWatchI18n;
  const tr=(source, values)=>I.t(source, values);
  if (root._fwCleanup) root._fwCleanup();
  if (!root._fwTemplate) root._fwTemplate=root.innerHTML;
  else root.innerHTML=root._fwTemplate;
  I.apply(root);
  const q = s => root.querySelector(s);
  const screen=q('.fw-workspace');
  if (!screen) return;
  if (!data || !data.cases || !data.cases.length) {
    screen.innerHTML='<div class="fw-error" role="status" aria-live="polite"></div>';
    screen.firstChild.textContent=tr(data?.message) || tr('Preparando a análise…');
    if(q('[data-credit]'))q('[data-credit]').textContent='';
    if(q('[data-overview]'))q('[data-overview]').hidden=true;
    if(q('.fw-result-details'))q('.fw-result-details').hidden=true;
    if(q('[data-scenario]'))q('[data-scenario]').hidden=true;
    const refreshEmpty=()=>{screen.firstChild.textContent=tr(data?.message || 'Preparando a análise…');};
    refreshEmpty();
    const unsubscribe=I.subscribe(refreshEmpty);
    root._fwCleanup=unsubscribe;
    return;
  }
  const video=q('[data-primary-video]'), select=q('[data-source]'), segments=q('[data-segments]');
  const motionVideo=q('[data-motion-video]'), motionCard=q('[data-motion-card]');
  const seek=q('[data-seek]'), start=q('[data-start]');
  const present=q('[data-present]');
  let current=null, mode='annotated', switching=false, frame=0, destroyed=false, generation=0;
  let motionFrames=[], motionFailed=false, motionPlayPending=false, motionAutoplayBlocked=false, motionKey='', motionPlayToken=0;
  let mainMedia=null, mainRequest=null, mainLoading=false, mainFailed=false, mainTarget=0, mainResume=false;
  let motionMedia=null, motionRequest=null, motionGeneration=0;
  const format=t=>Number(t||0).toLocaleString(I.locale,{minimumFractionDigits:1,maximumFractionDigits:1})+tr(' s');
  let names={none:tr('Não observada'),weak:tr('Baixa'),medium:tr('Moderada'),strong:tr('Alta')};
  let meanings={none:tr('O sistema não identificou sinais de atividade alimentar neste período.'),weak:tr('O sistema estimou pouca atividade alimentar neste período.'),medium:tr('O sistema estimou um nível intermediário de atividade alimentar neste período.'),strong:tr('O sistema estimou atividade alimentar intensa neste período.')};
  const setText=(selector,text)=>{const node=q(selector),value=tr(text);if(node && node.textContent!==value)node.textContent=value;};
  const decimal=n=>Number(n).toLocaleString(I.locale,{maximumFractionDigits:1});
  const activeSource=(element,media)=>Boolean(media && (element.currentSrc||element.src)===new URL(media.url,document.baseURI).href);
  function stopMotion(unload=false) {
    motionVideo.pause();
    if(unload){motionVideo.removeAttribute('src');motionVideo.load();motionMedia?.release();motionMedia=null;}
  }
  function syncMotion(force=false) {
    if(destroyed || !current?.motion?.video || motionFailed || motionVideo.readyState<1)return;
    const target=Math.max(0,Math.min(video.currentTime,Number.isFinite(motionVideo.duration)?motionVideo.duration:current.duration));
    if((force || Math.abs(motionVideo.currentTime-target)>.08) && !motionVideo.seeking) {
      try{motionVideo.currentTime=target;}catch{}
    }
    if(motionVideo.playbackRate!==video.playbackRate)motionVideo.playbackRate=video.playbackRate;
    if(video.paused || video.ended || switching){motionVideo.pause();return;}
    if(motionVideo.paused && !motionPlayPending && !motionAutoplayBlocked && (!motionVideo.ended || target<motionVideo.duration-.05)) {
      const token=++motionPlayToken;
      motionPlayPending=true;
      motionVideo.play().catch(error=>{if(token===motionPlayToken && error.name!=='AbortError')motionAutoplayBlocked=true;}).finally(()=>{if(token===motionPlayToken)motionPlayPending=false;});
    }
  }
  function configureMotion() {
    motionRequest?.abort();motionRequest=null;
    const version=++motionGeneration;
    stopMotion(true);
    motionPlayToken++;motionPlayPending=false;motionFailed=false;motionAutoplayBlocked=false;motionKey='';
    motionFrames=(current.motion?.frames||[]).filter(row=>Number.isFinite(row.timestamp_seconds));
    const available=Boolean(current.motion?.video);
    motionCard.dataset.available=String(available);
    motionCard.setAttribute('aria-busy',String(available));
    motionVideo.hidden=true;
    q('[data-motion-empty]').hidden=false;
    q('[data-motion-empty]').textContent=tr(available?'Carregando mapa…':'Mapa não disponível neste vídeo.');
    setText('[data-motion-status]',available?'Carregando mapa…':'Sem dados');
    q('[data-motion-legend]').hidden=!available;
    q('[data-motion-scale]').hidden=!available;
    q('[data-motion-metrics]').hidden=!motionFrames.length;
    if(available) {
      motionVideo.muted=true;motionVideo.defaultMuted=true;motionVideo.disablePictureInPicture=true;
      const controller=new AbortController();motionRequest=controller;
      resolveFeedWatchMedia(current.motion.video,{signal:controller.signal}).then(media=>{
        if(destroyed || version!==motionGeneration){media.release();return;}
        motionMedia=media;motionVideo.src=media.url;motionVideo.load();
      }).catch(error=>{
        if(destroyed || version!==motionGeneration || controller.signal.aborted)return;
        failMotion();
      }).finally(()=>{if(version===motionGeneration)motionRequest=null;});
    }
    updateMotionLabels();
  }
  function updateMotionLabels() {
    const scale=current?.motion?.scale_max_source_px_per_second;
    const threshold=current?.motion?.active_threshold_source_px_per_second;
    setText('[data-motion-scale]',Number.isFinite(scale)?'0–'+decimal(scale)+'+ px/s':'—');
    const description=Number.isFinite(threshold)?tr('Área em movimento: pixels acima de {threshold} px/s.',{threshold:decimal(threshold)}):'';
    q('[data-motion-threshold]').textContent=description;
    q('[data-motion-threshold]').hidden=!description;
    q('[data-motion-area]').title=description;
    motionVideo.setAttribute('aria-label',tr('Mapa de movimento sincronizado ao vídeo principal'));
    if(!current?.motion?.video)setText('[data-motion-status]','Sem dados');
    else if(motionFailed)setText('[data-motion-status]','Mapa indisponível.');
    else setText('[data-motion-status]',motionVideo.readyState<1?'Carregando mapa…':'Sincronizado');
  }
  function updateMotion(time) {
    syncMotion();
    if(!motionFrames.length)return;
    const key=Math.round(time*1000)+':'+I.language;
    if(key===motionKey)return;
    motionKey=key;
    const sample=motionFrameAt(motionFrames,time);
    const ready=sample?.status==='ok';
    const mean=ready && Number.isFinite(sample.mean_source_px_per_second) && sample.mean_source_px_per_second>=0?sample.mean_source_px_per_second:null;
    const area=ready && Number.isFinite(sample.moving_fraction) && sample.moving_fraction>=0 && sample.moving_fraction<=1?sample.moving_fraction:null;
    setText('[data-motion-mean]',mean===null?'—':decimal(mean)+' px/s');
    setText('[data-motion-area]',area===null?'—':decimal(100*area)+'%');
    if(current.motion?.video && !motionFailed && motionVideo.readyState>=1)setText('[data-motion-status]',sample?.status==='initializing'?'Preparando movimento':'Sincronizado');
    const chart=motionChartData(motionFrames,time,current.duration);
    q('[data-motion-path]').setAttribute('d',chart.path);
    q('[data-motion-cursor]').setAttribute('x1',chart.cursor.toFixed(2));
    q('[data-motion-cursor]').setAttribute('x2',chart.cursor.toFixed(2));
    setText('[data-motion-chart-max]','0–'+decimal(chart.maximum)+' px/s');
    setText('[data-motion-chart-time]',format(current.duration));
    q('[data-motion-chart]').setAttribute('aria-label',tr('Movimento médio da imagem até {seconds} s; escala de 0 a {maximum} px/s.',{seconds:decimal(time),maximum:decimal(chart.maximum)}));
  }
  motionVideo.onloadedmetadata=()=>{
    if(destroyed || !activeSource(motionVideo,motionMedia))return;
    motionVideo.hidden=false;q('[data-motion-empty]').hidden=true;
    motionCard.setAttribute('aria-busy','false');
    updateMotionLabels();syncMotion(true);
  };
  function failMotion() {
    motionFailed=true;stopMotion(true);motionVideo.hidden=true;motionCard.dataset.available='false';
    motionCard.setAttribute('aria-busy','false');
    q('[data-motion-empty]').hidden=false;q('[data-motion-empty]').textContent=tr('Mapa indisponível.');
    q('[data-motion-legend]').hidden=true;q('[data-motion-scale]').hidden=true;
    setText('[data-motion-status]','Mapa indisponível.');
  }
  motionVideo.onerror=()=>{
    if(destroyed || motionFailed || !activeSource(motionVideo,motionMedia))return;
    failMotion();
  };
  const scenarioForm=q('[data-scenario-form]');
  scenarioForm.oninput=()=>{q('[data-scenario-output]').hidden=true;};
  scenarioForm.onsubmit=event=>{
    event.preventDefault();
    const output=q('[data-scenario-output]');
    try {
      const price=q('[data-unit-price]').value;
      const result=calculateFeedingScenario(q('[data-usual]').valueAsNumber,q('[data-proposed]').valueAsNumber,price===''?null:q('[data-unit-price]').valueAsNumber);
      const quantity=n=>n.toLocaleString(I.locale,{maximumFractionDigits:3})+tr(' kg');
      const money=n=>n.toLocaleString(I.locale,{style:'currency',currency:'BRL'});
      const difference=result.difference_kg;
      let text=tr('{usual} habituais → {proposed} no cenário. ',{usual:quantity(result.usual_kg),proposed:quantity(result.proposed_kg)});
      text+=difference===0?tr('Sem alteração de quantidade.'):quantity(Math.abs(difference))+(difference>0?tr(' a menos'):tr(' a mais'))+(result.difference_percent===null?'':tr(' (')+decimal(Math.abs(result.difference_percent))+'%)')+'.';
      if(result.usual_cost!==null)text+=tr(' Custo: ')+money(result.usual_cost)+tr(' → ')+money(result.proposed_cost)+tr('. Diferença de ')+money(Math.abs(result.difference_cost))+(result.difference_cost===0?'.':result.difference_cost>0?tr(' a menos.'):tr(' a mais.'));
      output.textContent=text;
    } catch(error) { output.textContent=tr(error.message); }
    output.hidden=false;
  };
  function showOverview(overview) {
    q('[data-overview]').hidden=!overview;
    if(!overview)return;
    setText('[data-overview-intro]',tr('Duração: ')+decimal(overview.duration_seconds)+tr(' segundos.'));
    const winners=overview.predominant_classes;
    const summaryAvailable=overview.summary_available??(overview.coverage_fraction>=.8);
    setText('[data-overview-prediction]',!winners.length?tr('Sem resultado'):!summaryAvailable?tr('Dados insuficientes'):(overview.tie?tr('Sem predominância'):names[winners[0]]));
    setText('[data-overview-conclusion]',!winners.length?tr('Nenhum período teve uma classificação válida.'):!summaryAvailable?tr('É preciso classificar pelo menos 80% da duração para apresentar um nível predominante do vídeo.'):overview.tie?tr('O tempo analisado ficou empatado entre ')+winners.map(n=>names[n].toLowerCase()).join(tr(' e '))+'.':decimal(overview.seconds_by_class[winners[0]])+tr(' segundos com este nível, de ')+decimal(overview.analyzed_seconds)+tr(' segundos classificados.'));
    setText('[data-overview-coverage]',decimal(overview.analyzed_seconds)+tr(' de ')+decimal(overview.duration_seconds)+tr(' segundos tiveram classificação.')+(overview.unclassified_seconds>0.001?tr(' ')+decimal(overview.unclassified_seconds)+tr(' segundos ficaram sem resultado.'):''));
    setText('[data-overview-changes]',overview.adjacent_pairs?overview.changes+tr(' mudança')+(overview.changes===1?'':tr('s'))+tr(' de nível entre períodos consecutivos.'):tr('Este vídeo não tem períodos consecutivos suficientes para comparar mudanças.'));
    q('[data-distribution]').replaceChildren(...Object.keys(names).map(name=>{
      const row=document.createElement('div');row.className='fw-distribution-row';
      const label=document.createElement('span');label.textContent=names[name];
      const track=document.createElement('span');track.className='fw-distribution-track';track.setAttribute('aria-hidden','true');
      const bar=document.createElement('span');bar.dataset.class=name;bar.style.width=(100*overview.fractions_by_class[name])+'%';track.append(bar);
      const amount=document.createElement('span');amount.className='fw-distribution-value';amount.textContent=decimal(100*overview.fractions_by_class[name])+'%';
      amount.title=decimal(overview.seconds_by_class[name])+tr(' segundos');row.append(label,track,amount);return row;
    }));
    setText('[data-overview-count]',overview.detection_mean==null?tr('Sem dados'):decimal(overview.detection_mean)+tr(' em média'));
    setText('[data-overview-count-note]',overview.detection_mean==null?tr('Não há contagens disponíveis para calcular a média neste vídeo.'):tr('De ')+overview.detection_min+tr(' a ')+overview.detection_max+tr(' por imagem, em ')+overview.detection_frames+tr(' imagens com resultado.')+(overview.detection_missing_frames?tr(' ')+overview.detection_missing_frames+tr(' imagens ficaram sem contagem.'):'')+(overview.detection_zero_frames?tr(' Em ')+overview.detection_zero_frames+tr(' imagens, o detector não encontrou peixes.'):'')+tr(' A contagem pode ter erros e não representa o total do tanque.'));
    const evolution=overview.evolution, phases=q('[data-phases]');
    phases.replaceChildren();
    const short=evolution?.phases.some(p=>p.reason==='short_video');
    phases.hidden=!evolution || short;
    if(evolution && !short) {
      const titles={start:tr('Início'),middle:tr('Meio'),end:tr('Final')};
      phases.replaceChildren(...evolution.phases.map(phase=>{
        const item=document.createElement('li');
        const label=document.createElement('p');label.className='fw-label';label.textContent=titles[phase.key];
        const value=document.createElement('p');value.className='fw-phase-value';
        value.textContent=!phase.predominant_classes.length?tr('Sem resultado'):phase.predominant_classes.length>1?tr('Empate'):names[phase.predominant_classes[0]];
        const range=document.createElement('p');range.className='fw-phase-range';range.textContent=decimal(phase.start_seconds)+'–'+decimal(phase.end_seconds)+tr(' s');
        const coverage=document.createElement('p');coverage.className='fw-phase-coverage';coverage.textContent=decimal(phase.analyzed_seconds)+tr(' de ')+decimal(phase.end_seconds-phase.start_seconds)+tr(' s classificados');
        item.append(label,value,range,coverage);return item;
      }));
    }
    const comparison=evolution?.comparison;
    let conclusion=tr('A comparação entre partes não está disponível nesta análise.');
    if(short)conclusion=tr('A comparação entre início, meio e final precisa de pelo menos 12 segundos de vídeo.');
    else if(comparison?.available)conclusion=comparison.direction==='lower'?tr('No final, predominou um nível menor que no início.'):comparison.direction==='higher'?tr('No final, predominou um nível maior que no início.'):comparison.middle_differs?tr('Mesmo nível predominante no início e no final, com variação no meio.'):tr('Mesmo nível predominante nas três partes do vídeo.');
    else if(comparison?.reason==='ambiguous_predominance')conclusion=tr('Sem comparação conclusiva: pelo menos uma parte não teve um nível com mais da metade do tempo classificado.');
    else if(comparison)conclusion=tr('Sem dados suficientes para comparar as três partes. Há poucos resultados ou tempo sem classificação.');
    setText('[data-evolution]',conclusion);
  }
  if(data.live){setText('[data-demo-heading]',tr('Resultado do seu vídeo'));setText('[data-demo-intro]',tr('A classificação de cada período aparece durante a reprodução. O panorama abaixo resume o vídeo inteiro.'));}
  select.replaceChildren(...data.cases.map((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=tr(c.title);return o;}));
  function update() {
    if (!current) return;
    if(mainLoading){showMediaLoading();return;}
    if(mainFailed || video.error){setText('[data-prediction]',tr('Vídeo indisponível'));setText('[data-meaning]',tr('Selecione outro exemplo ou recarregue a página para tentar novamente.'));setText('[data-window]',tr('Selecione outro exemplo'));setText('[data-count]','—');start.textContent=tr('Vídeo indisponível');start.disabled=true;start.hidden=false;return;}
    const t=video.currentTime;
    updateMotion(t);
    const complete=current.windows.filter(w=>w.status!=='incomplete' && (w.available_at_seconds??w.end_seconds)<=t+1e-9);
    const latest=complete.at(-1), tail=current.windows.some(w=>w.status==='incomplete'&&t>=w.start_seconds);
    setText('[data-prediction]',latest ? (latest.status==='ok'?names[latest.predicted_class]:tr('Sem resultado')) : (t>0?tr('Aguardando'):tr('Inicie o vídeo')));
    setText('[data-meaning]',latest ? (latest.status==='ok'?meanings[latest.predicted_class]:tr('Não foi possível analisar este período do vídeo.')) : tr('O primeiro resultado aparece quando o vídeo completa ')+format(current.windows[0]?.end_seconds||2)+'.');
    setText('[data-window]',(latest ? format(latest.start_seconds)+tr(' a ')+format(latest.end_seconds) : tr('Resultado após ')+format(current.windows[0]?.end_seconds||2))+(tail?tr(' · Final sem nova análise'):''));
    const observation=current.frames?.[Math.min(Math.floor(t*current.fps),current.frames.length-1)];
    setText('[data-count]',current.hasCounts?(observation?.count==null?'—':observation.count.toLocaleString(I.locale)):tr('Sem dados'));
    setText('[data-clock]',format(t)+tr(' / ')+format(current.duration));
    seek.value=String(t);
    for (const button of segments.children) button.dataset.active=String(latest?.window_id===Number(button.dataset.id));
    start.hidden=!video.paused || t>0;
  }
  function showMediaLoading() {
    setText('[data-prediction]','Carregando vídeo…');setText('[data-window]','Preparando a reprodução');
    setText('[data-count]','—');setText('[data-motion-mean]','—');setText('[data-motion-area]','—');
    setText('[data-clock]','Carregando vídeo…');q('[data-motion-path]').setAttribute('d','');motionKey='';
    for(const button of segments.children)button.dataset.active='false';
    start.textContent=tr('Carregando vídeo…');start.disabled=true;start.hidden=false;seek.disabled=true;
  }
  async function changeMode(next, time=mainLoading?mainTarget:video.currentTime, playing=mainLoading?mainResume:!video.paused) {
    if (!current) return;
    switching=true; mode=next;
    const version=++generation;
    mainRequest?.abort();const controller=new AbortController();mainRequest=controller;
    mainLoading=true;mainFailed=false;mainTarget=time;mainResume=playing;
    q('[data-mode-label]').textContent=current.original===current.annotated?tr('Exemplo do conjunto de pesquisa'):(mode==='original'?tr('Gravação sem marcações'):tr('Peixes identificados pelo sistema'));
    for(const b of root.querySelectorAll('[data-mode]')) b.setAttribute('aria-pressed',String(b.dataset.mode===mode));
    video.pause();
    stopMotion();video.hidden=true;video.setAttribute('aria-busy','true');
    video.removeAttribute('src');video.load();mainMedia?.release();mainMedia=null;
    showMediaLoading();
    try {
      const media=await resolveFeedWatchMedia(current[mode],{signal:controller.signal});
      if(destroyed || version!==generation){media.release();return;}
      mainMedia=media;video.src=media.url;video.load();
    } catch(error) {
      if(destroyed || version!==generation || controller.signal.aborted)return;
      mainLoading=false;mainFailed=true;switching=false;video.setAttribute('aria-busy','false');update();
    } finally {if(version===generation)mainRequest=null;}
  }
  video.onloadedmetadata=()=>{
    if(destroyed || !activeSource(video,mainMedia))return;
    video.currentTime=Math.min(mainTarget,current.duration);mainLoading=false;switching=false;
    video.hidden=false;video.setAttribute('aria-busy','false');seek.disabled=false;
    start.disabled=false;start.textContent=tr('Reproduzir vídeo');motionKey='';update();syncMotion(true);
    if(mainResume)video.play().catch(()=>{});
  };
  function choose(index) {
    current=data.cases[index];
    configureMotion();
    showOverview(current.overview);
    q('.fw-mode').hidden=current.original===current.annotated;
    video.poster=current.poster||'';
    video.setAttribute('aria-label',tr(current.title)+tr(' — vídeo com controles'));
    q('[data-badge]').textContent=data.live?tr('Análise concluída'):tr('Exemplo já analisado');
    q('[data-scope]').textContent=tr(current.scope);
    q('[data-credit]').textContent=tr(current.credit);
    const hasCounts=current.frames?.some(f=>f.count!=null);
    current.hasCounts=hasCounts;
    q('[data-count]').dataset.unavailable=String(!hasCounts);
    q('[data-count-label]').textContent=tr('Peixes detectados');
    q('[data-count-note]').textContent=hasCounts?tr('Quantidade na imagem atual. Não representa o total de peixes do tanque.'):tr('Este vídeo não contém contagens de peixes disponíveis. A atividade alimentar é uma análise separada.');
    q('[data-reference]').hidden=!current.reference_class;
    q('[data-reference]').textContent=current.reference_class?tr('Referência da pesquisa: ')+names[current.reference_class]+tr('. Compare com a estimativa do sistema.'):'';
    seek.max=String(current.duration);
    segments.replaceChildren(...current.windows.map(w=>{
      const b=document.createElement('button');b.type='button';b.dataset.class=w.predicted_class||'unavailable';b.dataset.id=w.window_id;
      b.style.flex=String(w.end_seconds-w.start_seconds);
      b.textContent=(w.predicted_class==='none'?tr('Sem sinais'):names[w.predicted_class])||tr('Sem análise');
      b.title=format(w.start_seconds)+tr(' a ')+format(w.end_seconds)+tr(': ')+(names[w.predicted_class]||tr('Sem análise'))+(w.status==='ok'?tr('. Resultado disponível após ')+format(w.end_seconds):tr('. Período sem resultado'));
      b.setAttribute('aria-label',b.title);
      b.addEventListener('click',()=>{video.currentTime=Math.min(w.status==='ok'?w.end_seconds+0.001:w.start_seconds,current.duration);update();});
      return b;
    }));
    const coverage=current.windows.reduce((n,w)=>n+(w.status==='ok'?w.end_seconds-w.start_seconds:0),0);
    q('[data-coverage]').textContent=format(coverage)+tr(' analisados');
    changeMode(mode,0,false);
    const params=new URLSearchParams(location.search);params.set('exemplo',current.id);
    try{history.replaceState(null,'',location.pathname+'?'+params+location.hash);}catch{}
  }
  for(const b of root.querySelectorAll('[data-mode]')) b.onclick=()=>changeMode(b.dataset.mode);
  function updatePresentation() {
    const active=document.fullscreenElement===screen;
    present.textContent=active?tr('Sair da apresentação'):tr('Modo apresentação');
    present.setAttribute('aria-pressed',String(active));
    present.setAttribute('aria-label',active?tr('Sair da demonstração em tela cheia'):tr('Abrir demonstração em tela cheia'));
  }
  function onPresentationKey(event) {
    if(event.key==='Escape' && document.fullscreenElement===screen)document.exitFullscreen().catch(()=>{});
  }
  if(document.fullscreenEnabled && screen.requestFullscreen) {
    present.hidden=false;
    present.onclick=async()=>{
      q('[data-presentation-status]').hidden=true;
      try {
        if(document.fullscreenElement===screen)await document.exitFullscreen();
        else await screen.requestFullscreen();
      } catch {
        setText('[data-presentation-status]',tr('Não foi possível abrir a tela cheia. A reprodução continua disponível nesta página.'));
        q('[data-presentation-status]').hidden=false;
      }
    };
    document.addEventListener('fullscreenchange',updatePresentation);
    document.addEventListener('keydown',onPresentationKey);
  }
  select.onchange=()=>choose(Number(select.value));
  seek.oninput=()=>{video.currentTime=Number(seek.value);update();};
  start.onclick=()=>video.play().catch(()=>{});
  video.onplay=()=>{motionAutoplayBlocked=false;update();syncMotion(true);};
  video.onpause=()=>{stopMotion();update();syncMotion(true);};
  video.onended=()=>{stopMotion();update();syncMotion(true);};
  video.ontimeupdate=update;
  video.onseeking=()=>{stopMotion();syncMotion(true);update();};
  video.onseeked=()=>{syncMotion(true);update();};
  video.onratechange=()=>syncMotion(true);
  video.onerror=()=>{if(!activeSource(video,mainMedia))return;mainFailed=true;mainLoading=false;switching=false;video.setAttribute('aria-busy','false');stopMotion();update();};
  function tick(){if(destroyed)return;if(!switching)update();frame=requestAnimationFrame(tick);}
  const requested=new URLSearchParams(location.search).get('exemplo');
  const initial=Math.max(0,data.cases.findIndex(c=>c.id===requested));select.value=initial;choose(initial);tick();
  function refreshLanguage() {
    names={none:tr('Não observada'),weak:tr('Baixa'),medium:tr('Moderada'),strong:tr('Alta')};
    meanings={none:tr('O sistema não identificou sinais de atividade alimentar neste período.'),weak:tr('O sistema estimou pouca atividade alimentar neste período.'),medium:tr('O sistema estimou um nível intermediário de atividade alimentar neste período.'),strong:tr('O sistema estimou atividade alimentar intensa neste período.')};
    for(const [index,option] of [...select.options].entries())option.textContent=tr(data.cases[index].title);
    if(current) {
      showOverview(current.overview);
      video.setAttribute('aria-label',tr(current.title)+tr(' — vídeo com controles'));
      q('[data-badge]').textContent=tr(data.live?'Análise concluída':'Exemplo já analisado');
      q('[data-scope]').textContent=tr(current.scope);
      q('[data-credit]').textContent=tr(current.credit);
      q('[data-count-label]').textContent=tr('Peixes detectados');
      q('[data-count-note]').textContent=tr(current.hasCounts?'Quantidade na imagem atual. Não representa o total de peixes do tanque.':'Este vídeo não contém contagens de peixes disponíveis. A atividade alimentar é uma análise separada.');
      q('[data-reference]').textContent=current.reference_class?tr('Referência da pesquisa: ')+names[current.reference_class]+tr('. Compare com a estimativa do sistema.'):'';
      q('[data-mode-label]').textContent=tr(current.original===current.annotated?'Exemplo do conjunto de pesquisa':(mode==='original'?'Gravação sem marcações':'Peixes identificados pelo sistema'));
      for(const [index,button] of [...segments.children].entries()) {
        const w=current.windows[index];
        button.textContent=(w.predicted_class==='none'?tr('Sem sinais'):names[w.predicted_class])||tr('Sem análise');
        button.title=format(w.start_seconds)+tr(' a ')+format(w.end_seconds)+': '+(names[w.predicted_class]||tr('Sem análise'))+(w.status==='ok'?tr('. Resultado disponível após ')+format(w.end_seconds):tr('. Período sem resultado'));
        button.setAttribute('aria-label',button.title);
      }
      const coverage=current.windows.reduce((n,w)=>n+(w.status==='ok'?w.end_seconds-w.start_seconds:0),0);
      q('[data-coverage]').textContent=format(coverage)+tr(' analisados');
      motionKey='';updateMotionLabels();
      q('[data-motion-empty]').textContent=tr(motionFailed?'Mapa indisponível.':motionRequest?'Carregando mapa…':'Mapa não disponível neste vídeo.');
    }
    if(data.live){setText('[data-demo-heading]','Resultado do seu vídeo');setText('[data-demo-intro]','A classificação de cada período aparece durante a reprodução. O panorama abaixo resume o vídeo inteiro.');}
    if(!q('[data-scenario-output]').hidden)scenarioForm.onsubmit({preventDefault(){}});
    updatePresentation();
    update();
  }
  const unsubscribe=I.subscribe(refreshLanguage);
  root._fwCleanup=()=>{unsubscribe();destroyed=true;generation++;motionGeneration++;mainRequest?.abort();motionRequest?.abort();cancelAnimationFrame(frame);video.pause();video.removeAttribute('src');video.load();mainMedia?.release();mainMedia=null;stopMotion(true);document.removeEventListener('fullscreenchange',updatePresentation);document.removeEventListener('keydown',onPresentationKey);};
}
