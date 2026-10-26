"""Compact presentation: scene classification, detector observations and image motion."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .temporal_examples import effective_crop_box
from .video_overlay import font_file, interval


class DynamicOverlay:
    def __init__(self, width, height, predictor, config):
        self.width, self.height, self.config = width, height, config
        self.scale = max(.5, width/1920)
        self.roi, self.crop = effective_crop_box(width, height, predictor.extraction, predictor.transform)
        self.fonts = {size: ImageFont.truetype(font_file(size == 23), max(10, round(size*self.scale)))
                      for size in (12, 14, 16, 23)}
        self.history = []

    def render(self, source, food, mode, observations, detector_status, motion, flow, timestamp, clip_context=""):
        # Work on a copy; visualization must never become model input.
        frame, cfg, s = source.copy(), self.config["motion"], self.scale
        if flow is not None:
            mag = np.linalg.norm(flow, axis=2)
            display = np.clip(mag/cfg["heatmap_max_source_px_per_second"], 0, 1)
            heat = cv2.applyColorMap(np.uint8(display*255), cv2.COLORMAP_OCEAN)
            heat = cv2.resize(heat, (self.width, self.height))
            alpha = cv2.resize(display, (self.width, self.height))[:, :, None]*cfg["heatmap_opacity"]
            frame = np.uint8(frame*(1-alpha)+heat*alpha)
            step = cfg["vector_spacing_analysis_pixels"]
            for y in range(step//2, flow.shape[0], step):
                for x in range(step//2, flow.shape[1], step):
                    velocity = flow[y, x]
                    if np.linalg.norm(velocity) <= cfg["active_threshold_source_px_per_second"]:
                        continue
                    origin = np.array([x*self.width/flow.shape[1], y*self.height/flow.shape[0]])
                    # Length is 0.10 s displacement, capped only for display readability.
                    vector = velocity*.10
                    length = np.linalg.norm(vector)
                    if length > 36*s:
                        vector *= 36*s/length
                    cv2.arrowedLine(frame, tuple(np.int32(origin)), tuple(np.int32(origin+vector)),
                        (220, 190, 65), max(1, round(s)), cv2.LINE_AA, tipLength=.25)
        canvas = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        layer = Image.new("RGBA", canvas.size)
        draw = ImageDraw.Draw(layer)
        pad, line = round(12*s), max(1, round(1.4*s))
        def text(x, y, value, size=14, color=(237, 247, 247, 255)):
            draw.text((round(x), round(y)), value, font=self.fonts[size], fill=color)
        def panel(box):
            draw.rounded_rectangle(tuple(round(v) for v in box), radius=round(6*s), fill=(9, 26, 32, 208))
        # Subtle corners indicate the exact GRU input geometry; not a detection box.
        l, t, r, b = self.crop
        corner = 15*s
        for x, y, dx, dy in ((l,t,1,1), (r,t,-1,1), (l,b,1,-1), (r,b,-1,-1)):
            draw.line([(x+dx*corner,y),(x,y),(x,y+dy*corner)], fill=(222,235,235,125), width=line)
        text(l+5*s, t+3*s, "recorte GRU", 12, (225,235,235,165))
        for obs in observations:
            # Display ONLY current, real detector observations, never predicted missing boxes.
            color = (150, 238, 163, 240)
            trail = [(point[1], point[2]) for point in obs["trail_txy"]]
            if len(trail) >= 2:
                draw.line(trail, fill=(150,238,163,180), width=line)
            x1, y1, x2, y2 = obs["xyxy"]
            draw.rectangle((x1,y1,x2,y2), outline=color, width=line)
            speed = obs["speed_source_px_per_second"]
            label = f'#{obs["track_id"]}'+(f" · {speed:.0f} px/s" if speed is not None else "")
            box = draw.textbbox((0,0), label, font=self.fonts[12])
            tx, ty = min(x1, self.width-box[2]-6*s), max(0, y1-20*s)
            draw.rectangle((tx,ty,tx+box[2]+5*s,ty+20*s), fill=(9,26,32,200))
            text(tx+2*s,ty,label,12,color)
        panel((pad,pad,pad+590*s,pad+92*s))
        item = food.get("window")
        prediction = (f'{item["predicted_class"]} · {item["legend_pt"]}' if food["status"] == "prediction"
                      else "Analisando" if food["status"] == "analyzing" else "Análise indisponível")
        text(pad+12*s,pad+7*s,"FeedWatch  |  Cena: "+prediction,23)
        description = "Análise do clipe completo" if mode == "clip" else "Janelas consecutivas · experimental"
        text(pad+12*s,pad+40*s,description,14)
        text(pad+12*s,pad+63*s,("Intervalo "+interval(item) if item else "Primeira janela em análise")
             +(" · final incompleto, última previsão" if food["tail"] else ""),12)
        right = self.width-pad-395*s
        panel((right,pad,self.width-pad,pad+69*s))
        detection_text = (f"Peixes detectados no frame: {len(observations)}" if detector_status == "ok"
                          else "Detector desligado" if detector_status == "disabled" else "Detecção indisponível")
        text(right+12*s,pad+7*s,detection_text,16,(171,241,182,255))
        text(right+12*s,pad+34*s,"Pré-treinado · IDs temporários · px/s aparente",12)
        if clip_context:
            # Always mark which independent clip is being played.
            text(pad,self.height-38*s,clip_context,14)
        mean = motion.get("mean_source_px_per_second") if motion["status"] == "ok" else None
        self.history.append((timestamp, mean))
        self.history = [(t,v) for t,v in self.history if timestamp-t <= cfg["graph_history_seconds"]]
        bottom = self.height-50*s
        top, left, width = bottom-108*s, pad, 370*s
        panel((left,top,left+width,bottom))
        value = f"{mean:.1f} px/s" if mean is not None else "inicializando" if motion["status"] == "initializing" else "indisponível"
        text(left+10*s,top+5*s,"Movimento da imagem: "+value,14,(105,218,239,255))
        text(left+10*s,top+29*s,"Inclui água e reflexos · não é saída da GRU",12)
        graph_top, graph_bottom = top+56*s, bottom-10*s
        graph_width = width-20*s
        draw.line((left+10*s,graph_bottom,left+width-10*s,graph_bottom),fill=(128,151,160,190),width=1)
        valid = [v for _,v in self.history if v is not None]
        ceiling = max(20., max(valid,default=20.)*1.1)  # past/current values only, display scale only
        chain = []
        for t, v in self.history:
            if v is None:
                if len(chain)>1:
                    draw.line(chain,fill=(105,218,239,255),width=line)
                chain=[]
                continue
            x = left+10*s+graph_width*(t-max(0.,timestamp-cfg["graph_history_seconds"]))/cfg["graph_history_seconds"]
            y = graph_bottom-(graph_bottom-graph_top)*min(1.,v/ceiling)
            chain.append((x,y))
        if len(chain)>1:
            draw.line(chain,fill=(105,218,239,255),width=line)
        text(self.width-605*s,self.height-27*s,"Intensidade alimentar observada — protótipo experimental",12)
        return cv2.cvtColor(np.asarray(Image.alpha_composite(canvas,layer).convert("RGB")),cv2.COLOR_RGB2BGR)


def transition_image(width, height, number, total, relative_path, dataset_class):
    canvas = Image.new("RGB",(width,height),(12,30,37))
    draw = ImageDraw.Draw(canvas)
    scale = width/1920
    def text(y,value,size,color):
        font = ImageFont.truetype(font_file(size>=38),round(size*scale))
        draw.text((width*.09,height*y),value,font=font,fill=color)
    text(.25,"FeedWatch · montagem de clipes independentes",30,(153,204,213))
    text(.36,f"Clipe {number:02d}/{total:02d} · rastreamento reiniciado",46,(238,247,245))
    text(.49,relative_path,24,(180,205,213))
    text(.58,f"Rótulo do dataset: {dataset_class}  |  não usado na inferência",26,(180,205,213))
    text(.72,"Corte entre clipes · não representa uma sessão contínua",26,(153,204,213))
    return cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR)
