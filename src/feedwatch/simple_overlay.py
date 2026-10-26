"""Mobile-readable overlay. Whole source image is preserved above an added status footer."""

import cv2
import math
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from .temporal_examples import effective_crop_box
from .video_overlay import font_file,interval


class SimpleOverlay:
    def __init__(self,width,height,predictor,config,diagnostic=False,source_note=""):
        self.width,self.height,self.config=width,height,config
        self.diagnostic,self.source_note=diagnostic,source_note
        self.footer=round(width*config['presentation']['footer_fraction_of_width'])
        self.footer+=self.footer%2
        self.roi,self.crop=effective_crop_box(width,height,predictor.extraction,predictor.transform)
        self.fonts={name:ImageFont.truetype(font_file(True),max(12,round(width*factor)))
                    for name,factor in [('main',.036),('count',.032),('small',.026),('id',.024),('speed',.022)]}
        self.footer=max(self.footer,math.ceil(width*.087+self.fonts['count'].getbbox('Peixes detectados neste frame: 99')[3]+width*.012))
        self.footer+=self.footer%2
        self.last_layout={}

    def output_metadata(self,metadata):
        return {**metadata,'height':self.height+self.footer}

    def render(self,source,food,mode,observations,detector_status,motion,flow,timestamp,clip_context=""):
        frame=source.copy()
        if self.diagnostic and flow is not None:
            for y in range(24,flow.shape[0],48):
                for x in range(24,flow.shape[1],48):
                    v=flow[y,x]*.1
                    length=np.linalg.norm(v)
                    if length<2:continue
                    if length>35:v*=35/length
                    p=np.array([x*self.width/flow.shape[1],y*self.height/flow.shape[0]])
                    cv2.arrowedLine(frame,tuple(np.int32(p)),tuple(np.int32(p+v)),(240,200,30),2,tipLength=.3)
        canvas=Image.new('RGB',(self.width,self.height+self.footer),(13,32,40))
        canvas.paste(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),(0,0))
        draw=ImageDraw.Draw(canvas)
        stroke=max(2,round(self.width*.0015))
        color=(255,224,60)
        occupied=[]
        omitted=0
        def text(x,y,value,name='small',fill=(243,248,247),outline=False):
            draw.text((round(x),round(y)),value,font=self.fonts[name],fill=fill,
                stroke_width=stroke if outline else 0,stroke_fill=(7,26,32))
        def intersects(a,b):
            return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])
        # All boxes stay visible. Labels are placed separately so their text does not cover boxes or other labels.
        obstacle_boxes=[o['xyxy'] for o in observations]
        for obs in observations:
            x1,y1,x2,y2=obs['xyxy']
            points=[(p[1],p[2]) for p in obs['trail_txy']]
            if len(points)>=2:
                draw.line(points,fill=(30,50,55),width=stroke+2)
                draw.line(points,fill=color,width=max(2,stroke-1))
            # Outline only; neither fish nor water is filled in.
            draw.rectangle((x1,y1,x2,y2),outline=(10,32,40),width=stroke+3)
            draw.rectangle((x1,y1,x2,y2),outline=color,width=stroke)
        label_boxes=[]
        speed_labels=0
        for obs in sorted(observations,key=lambda o:(-o.get('consecutive_observations',0),o['track_id'])):
            if not obs.get('track_confirmed',False):continue
            x1,y1,x2,y2=obs['xyxy'];speed=obs['speed_source_px_per_second']
            show_speed=self.config['presentation'].get('show_speed',False)
            options=([f'#{obs["track_id"]} · {speed:.0f} px/s'] if show_speed and speed is not None else [])+[f'#{obs["track_id"]}']
            selected=None
            gap=stroke+4
            for label in options:
                bounds=draw.textbbox((0,0),label,font=self.fonts['id'],stroke_width=stroke,anchor='lt')
                tw,th=bounds[2]-bounds[0],bounds[3]-bounds[1]
                candidates=[(x1,y1-th-gap),(x2-tw,y1-th-gap),(x2+gap,y1),
                    (x1-tw-gap,y1),(x1,y2+gap),(x2-tw,y2+gap)]
                for x,y in candidates:
                    rect=(x-stroke,y-stroke,x+tw,y+th)
                    if (rect[0]<0 or rect[1]<0 or rect[2]>self.width or rect[3]>self.height
                        or any(intersects(rect,b) for b in occupied+obstacle_boxes)):
                        continue
                    selected=(x,y,label,rect);break
                if selected is not None:break
            if selected is None:
                omitted+=1
            else:
                x,y,label,rect=selected
                occupied.append(rect);label_boxes.append(rect)
                speed_labels+= 'px/s' in label
                draw.text((round(x),round(y)),label,font=self.fonts['id'],fill=color,
                    stroke_width=stroke,stroke_fill=(7,26,32),anchor='lt')
        if self.diagnostic:
            l,t,r,b=self.crop
            draw.rectangle((l,t,r,b),outline=(200,214,216),width=2)
            text(12,12,'Diagnóstico: recorte GRU; setas = movimento da imagem','speed',(240,245,245),True)
            if motion.get('status')=='ok':
                text(12,self.height-self.width*.04,f'Movimento da imagem: {motion["mean_source_px_per_second"]:.1f} px/s','speed',(160,231,244),True)
        item=food.get('window')
        value=(item['legend_pt'] if food['status']=='prediction' else 'Analisando' if food['status']=='analyzing' else 'Indisponível')
        x=self.width*.018;y=self.height+self.width*.009
        text(x,y,'Intensidade alimentar prevista: '+value,'main')
        origin=('Intervalo analisado: '+interval(item) if item else 'Aguardando a primeira janela completa')
        if mode=='clip':origin='Análise do clipe completo · '+(interval(item) if item else '')
        if food['tail']:origin+=' · final incompleto'
        text(x,y+self.width*.043,origin,'small',(192,212,220))
        count=(str(len(observations)) if detector_status=='ok' else 'indisponível' if detector_status=='unavailable' else 'detector desligado')
        text(x,y+self.width*.078,'Peixes detectados neste frame: '+count,'count',color)
        # Short, explicitly exploratory marker, leaving the classification sentence unambiguous.
        if self.source_note:
            text(self.width*.70,y+self.width*.045,'FeedWatch · Exploratório','speed',(192,212,220))
        self.last_layout={'omitted_id_labels_due_overlap':omitted,'footer_added_pixels':self.footer,
                          'source_image_scaled':False,'diagnostic':self.diagnostic,
                          'displayed_speed_labels':speed_labels,'label_boxes_xyxy':label_boxes,
                          'footer_text_bottom':draw.textbbox((x,y+self.width*.078),'Peixes detectados neste frame: '+count,font=self.fonts['count'])[3]}
        if self.last_layout['footer_text_bottom']>canvas.height-2:raise ValueError('Texto do rodapé ultrapassa a imagem')
        return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
