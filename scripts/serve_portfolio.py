"""Local preview with HTTP byte-range support for faithful MP4 seeking."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

app=FastAPI(docs_url=None,redoc_url=None)
app.mount('/',StaticFiles(directory=Path(__file__).resolve().parents[1]/'sites/feedwatch/dist',html=True),name='portfolio')

if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=8000,log_level='warning')
