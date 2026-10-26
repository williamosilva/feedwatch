"""Plain deterministic reporting and an entirely independent arithmetic simulator."""

from decimal import Decimal, InvalidOperation

from .video_inference import PORTUGUESE


def plain_summary(summary,analysis):
    valid=[w for w in analysis['windows'] if w['status']=='ok']
    changes=sum(a['predicted_class']!=b['predicted_class'] for a,b in zip(valid,valid[1:])
                if a['end_seconds']==b['start_seconds'])
    count=summary['complete_valid_windows']
    names={0:'zero',1:'um',2:'dois',3:'três',4:'quatro',5:'cinco',6:'seis'}
    parts=[f'**Previsões:** Foram analisados {names.get(count,str(count))} trechos completos de dois segundos '
           f'({summary["valid_analyzed_seconds"]:g} s de um vídeo de {summary["duration_seconds"]:g} s).']
    distribution='; '.join(f'{PORTUGUESE[c].lower()}: {n}' for c,n in summary['class_distribution'].items() if n)
    if valid:
        change_text=(f' Houve {changes} mudança(s) entre trechos consecutivos.' if changes else
                     ' A previsão foi a mesma nos trechos válidos.' if len({w['predicted_class'] for w in valid})==1
                     else ' As classes diferentes ocorreram em trechos separados por falhas.')
        parts.append('Previsões por trecho: '+distribution+'. '+summary['conclusion']+change_text)
    else:parts.append('Análise alimentar indisponível; nenhuma classe foi atribuída aos trechos inválidos.')
    low,high=summary['detections_per_frame_min'],summary['detections_per_frame_max']
    measure=(f'O detector encontrou entre {low} e {high} peixes por imagem do vídeo. '
             'Essa faixa inclui possíveis erros e não é o total de indivíduos do tanque.' if low is not None
             else 'As detecções ficaram indisponíveis.')
    if summary['frames_with_suspended_tracks']:
        frequency='frequentes' if summary['frames_with_suspended_tracks']/max(1,analysis['rendered_frames'])>=.2 else 'pontuais'
        measure+=f' Houve perdas {frequency} de trilhas; os IDs são temporários e podem mudar.'
    parts.append('**Medidas:** '+measure)
    failures=len(summary['invalid_windows'])
    if failures or summary['frames_with_visual_errors']:
        parts.append(f'Cobertura: {count}/{summary["window_count"]} trechos válidos; '
                     f'{failures} trecho(s) incompleto(s)/com falha; {summary["frames_with_visual_errors"]} imagens com falha visual. '
                     'Trechos incompletos não recebem preenchimento nem classificação.')
    parts.append('**Interpretação:** Aplicação exploratória fora das condições avaliadas, sem rótulos verificados por trecho. '
                 'Previsões ausente ou fraca não provam que os peixes não se alimentam. '
                 'O resultado descreve a intensidade prevista; não comprova fome ou saciedade nem orienta o fornecimento de ração.')
    return '\n\n'.join(parts)


def simulate_costs(price,package_kg,planned_kg,reduction_percent):
    if any(x is None for x in (price,package_kg,planned_kg,reduction_percent)):
        raise ValueError('Preencha os quatro valores para simular; eles não são necessários para analisar vídeos.')
    try:values=[Decimal(str(x)) for x in (price,package_kg,planned_kg,reduction_percent)]
    except InvalidOperation as error:raise ValueError('Informe números válidos') from error
    price,package_kg,planned_kg,reduction=values
    if any(not x.is_finite() for x in values):raise ValueError('Informe números finitos')
    if price<0 or package_kg<=0 or planned_kg<0 or not 0<=reduction<=100:
        raise ValueError('Preço e quantidade devem ser não negativos, peso maior que zero e percentual entre 0 e 100.')
    unit=price/package_kg;planned=unit*planned_kg;difference_kg=planned_kg*reduction/100
    return {'cost_per_kg':float(unit),'planned_cost':float(planned),'hypothetical_difference_kg':float(difference_kg),
            'hypothetical_difference_cost':float(difference_kg*unit),'user_reduction_percent':float(reduction)}


def cost_text(*values):
    data=simulate_costs(*values)
    def brl(n):return 'R$ '+f'{n:,.2f}'.replace(',','X').replace('.',',').replace('X','.')
    return (f'Custo por kg: **{brl(data["cost_per_kg"])}**. Custo planejado: **{brl(data["planned_cost"])}**. '
            f'Na hipótese de {data["user_reduction_percent"]:g}% informada por você, a diferença seria de '
            f'**{data["hypothetical_difference_kg"]:g} kg / {brl(data["hypothetical_difference_cost"])}**.\n\n'
            'Simulação aritmética independente do vídeo. Não é economia demonstrada nem recomendação de dosagem ou nutrição.')
