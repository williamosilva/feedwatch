"""Whole-video descriptions of recorded outputs, independent of playback position.

Durations summarize predicted categories, never hunger, accuracy or confidence.
"""
import math

CLASSES=('none','weak','medium','strong')

# Descriptive display rules, not validated biological decision thresholds.
SUMMARY_POLICY={'min_coverage_fraction':.8}
EVOLUTION_POLICY={'min_phase_seconds':4.,'min_windows_per_phase':2,'min_coverage_fraction':.8,
                  'require_majority':True}


def _evolution(valid, duration):
    phases=[]
    for i,key in enumerate(('start','middle','end')):
        start,end=duration*i/3,duration*(i+1)/3
        seconds={name:0. for name in CLASSES}
        contributors=0
        for a,b,label in valid:
            overlap=max(0.,min(b,end)-max(a,start))
            if overlap>1e-9:
                seconds[label]+=overlap
                contributors+=1
        covered=sum(seconds.values())
        winners=[name for name in CLASSES if covered and math.isclose(seconds[name],max(seconds.values()),abs_tol=1e-8)]
        coverage=covered/(end-start)
        if end-start<EVOLUTION_POLICY['min_phase_seconds']-1e-9:reason='short_video'
        elif contributors<EVOLUTION_POLICY['min_windows_per_phase']:reason='too_few_windows'
        elif coverage<EVOLUTION_POLICY['min_coverage_fraction']-1e-9:reason='missing_results'
        elif len(winners)!=1 or seconds[winners[0]]<=covered/2+1e-9:reason='ambiguous_predominance'
        else:reason=None
        phases.append({'key':key,'start_seconds':start,'end_seconds':end,'analyzed_seconds':covered,
                       'unclassified_seconds':max(0.,end-start-covered),'coverage_fraction':coverage,
                       'seconds_by_class':seconds,'fractions_by_class':{n:seconds[n]/covered if covered else 0. for n in CLASSES},
                       'predominant_classes':winners,'contributing_windows':contributors,
                       'comparable':reason is None,'reason':reason})
    comparable=all(p['comparable'] for p in phases)
    comparison={'available':comparable,'direction':None,'reason':next((p['reason'] for p in phases if p['reason']),None),
                'middle_differs':None}
    if comparable:
        first,middle,last=(p['predominant_classes'][0] for p in phases)
        comparison['direction']='lower' if CLASSES.index(last)<CLASSES.index(first) else 'higher' if CLASSES.index(last)>CLASSES.index(first) else 'same'
        comparison['middle_differs']=first==last and middle!=first
    return {'method':'three_equal_time_parts_v1','policy':dict(EVOLUTION_POLICY),'phases':phases,'comparison':comparison}


def summarize_video(windows, duration_seconds, frames=()):
    duration=float(duration_seconds)
    if not math.isfinite(duration) or duration<=0:raise ValueError('Duração inválida')
    seconds={name:0. for name in CLASSES}
    valid=[]
    for window in windows:
        if window.get('status')!='ok':continue
        start,end=float(window['start_seconds']),float(window['end_seconds'])
        label=window['predicted_class']
        if label not in seconds or not all(math.isfinite(v) for v in (start,end)) or start<0 or end<=start or end>duration+1e-6:
            raise ValueError('Período válido com classe ou limites inválidos')
        valid.append((start,min(end,duration),label))
    valid.sort()
    for i,(start,end,label) in enumerate(valid):
        if i and start<valid[i-1][1]-1e-6:raise ValueError('Períodos sobrepostos')
        seconds[label]+=end-start
    covered=sum(seconds.values())
    winners=[name for name in CLASSES if covered and math.isclose(seconds[name],max(seconds.values()),abs_tol=1e-8)]
    summary_available=bool(valid) and covered/duration>=SUMMARY_POLICY['min_coverage_fraction']-1e-9
    pairs=[(a,b) for a,b in zip(valid,valid[1:]) if math.isclose(a[1],b[0],abs_tol=1e-6)]
    changes=sum(a[2]!=b[2] for a,b in pairs)
    counts=[]
    for frame in frames:
        count=frame.get('count')
        if count is None:continue
        if isinstance(count,bool) or not isinstance(count,(int,float)) or not math.isfinite(count) or count<0 or int(count)!=count:
            raise ValueError('Contagem por imagem inválida')
        counts.append(count)
    return {'method':'predicted_duration_distribution_v1','duration_seconds':duration,
        'analyzed_seconds':covered,'unclassified_seconds':max(0.,duration-covered),
        'coverage_fraction':covered/duration,'seconds_by_class':seconds,
        'fractions_by_class':{name:seconds[name]/covered if covered else 0. for name in CLASSES},
        'predominant_classes':winners,'tie':len(winners)>1,'valid_windows':len(valid),
        'summary_policy':dict(SUMMARY_POLICY),'summary_available':summary_available,
        'summary_reason':None if summary_available else 'no_results' if not valid else 'insufficient_coverage',
        'evolution':_evolution(valid,duration),
        'adjacent_pairs':len(pairs),'changes':changes,'change_fraction':changes/len(pairs) if pairs else None,
        'detection_mean':sum(counts)/len(counts) if counts else None,
        'detection_min':min(counts) if counts else None,'detection_max':max(counts) if counts else None,
        'detection_frames':len(counts),'detection_missing_frames':len(frames)-len(counts),
        'detection_zero_frames':sum(count==0 for count in counts),
        'detection_coverage_fraction':len(counts)/len(frames) if frames else None,
        'scope':'Saved predictions for the entire processed video; time shares are not accuracy or calibrated confidence; counts are per image, not unique fish.'}
