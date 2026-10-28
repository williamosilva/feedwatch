"""Streaming hosts keep their URL; whole responses become one owned seekable Blob."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('node'), 'Node.js is required for media transport checks')
class MediaResolverUiTest(unittest.TestCase):
    def evaluate(self, expression):
        source = Path('src/feedwatch/assets/portfolio.js').read_text(encoding='utf-8')
        code = source + '\n(async()=>(' + expression + '))().then(value=>process.stdout.write(JSON.stringify(value))).catch(error=>{console.error(error);process.exit(1);});'
        result = subprocess.run(['node'], input=code, encoding='utf-8', capture_output=True, check=True)
        return json.loads(result.stdout)

    def test_valid_partial_response_keeps_streaming_and_cancels_only_probe(self):
        result = self.evaluate("(async()=>{let fetches=0,cancelled=0,created=0,revoked=0,range='';const media=await resolveFeedWatchMedia('/video.mp4',{fetchImpl:async(url,options)=>{fetches++;range=options.headers.Range;return {status:206,headers:{get:()=> 'bytes 0-0/1234'},body:{cancel:async()=>cancelled++},blob:async()=>{throw Error('Should not read full media');}};},urlApi:{createObjectURL:()=>created++,revokeObjectURL:()=>revoked++}});media.release();return {url:media.url,fetches,cancelled,created,revoked,range};})()")
        self.assertEqual(result, {'url': '/video.mp4', 'fetches': 1, 'cancelled': 1, 'created': 0, 'revoked': 0, 'range': 'bytes=0-0'})

    def test_whole_response_is_consumed_once_and_released_once(self):
        result = self.evaluate("(async()=>{let fetches=0,reads=0,created=0;const revoked=[];const media=await resolveFeedWatchMedia('/video.mp4',{fetchImpl:async()=>{fetches++;return {status:200,headers:{get:()=> 'video/mp4'},blob:async()=>{reads++;return new Blob(['media'],{type:'video/mp4'});}};},urlApi:{createObjectURL:()=>{created++;return 'blob:owned';},revokeObjectURL:url=>revoked.push(url)}});media.release();media.release();return {url:media.url,fetches,reads,created,revoked};})()")
        self.assertEqual(result, {'url': 'blob:owned', 'fetches': 1, 'reads': 1, 'created': 1, 'revoked': ['blob:owned']})

    def test_received_blob_and_external_blob_url_do_not_probe(self):
        result = self.evaluate("(async()=>{let fetched=0,created=0;const revoked=[];const options={fetchImpl:async()=>fetched++,urlApi:{createObjectURL:()=>{created++;return 'blob:own';},revokeObjectURL:url=>revoked.push(url)}};const local=await resolveFeedWatchMedia(new Blob(['media']),options);local.release();const external=await resolveFeedWatchMedia('blob:external',options);external.release();return {fetched,created,revoked,external:external.url};})()")
        self.assertEqual(result, {'fetched': 0, 'created': 1, 'revoked': ['blob:own'], 'external': 'blob:external'})

    def test_http_failure_and_malformed_partial_response_are_not_hidden(self):
        result = self.evaluate("(async()=>{const output=[];for(const status of [404,206]){try{await resolveFeedWatchMedia('/video.mp4',{fetchImpl:async()=>({status,headers:{get:()=> 'bytes 1-9/100'},body:{cancel:async()=>{}}})});output.push(false);}catch(error){output.push(true);}}return output;})()")
        self.assertEqual(result, [True, True])

    def test_cross_origin_cors_failure_falls_back_but_local_network_failure_does_not(self):
        result = self.evaluate("(async()=>{const options={baseUrl:'https://local.test/',fetchImpl:async()=>{throw new TypeError('Failed to fetch');}};const external=await resolveFeedWatchMedia('https://other.test/video.mp4',options);let localFailed=false;try{await resolveFeedWatchMedia('/video.mp4',options);}catch(error){localFailed=true;}return {url:external.url,localFailed};})()")
        self.assertEqual(result, {'url': 'https://other.test/video.mp4', 'localFailed': True})

    def test_aborted_request_does_not_fall_back_to_external_url(self):
        result = self.evaluate("(async()=>{const controller=new AbortController();let created=0;try{await resolveFeedWatchMedia('https://other.test/video.mp4',{signal:controller.signal,fetchImpl:async()=>{controller.abort();throw new TypeError('Aborted fetch');},urlApi:{createObjectURL:()=>created++,revokeObjectURL:()=>{}}});return {error:null,created};}catch(error){return {error:error.name,created};}})()")
        self.assertEqual(result, {'error': 'AbortError', 'created': 0})

    def test_abort_while_reading_body_does_not_create_blob_url(self):
        result = self.evaluate("(async()=>{const controller=new AbortController();let created=0;try{await resolveFeedWatchMedia('/video.mp4',{signal:controller.signal,fetchImpl:async()=>({status:200,headers:{get:()=> 'video/mp4'},blob:async()=>{controller.abort();return new Blob(['media']);}}),urlApi:{createObjectURL:()=>created++,revokeObjectURL:()=>{}}});return {error:null,created};}catch(error){return {error:error.name,created};}})()")
        self.assertEqual(result, {'error': 'AbortError', 'created': 0})

    def test_abort_during_url_creation_releases_new_owned_url(self):
        result = self.evaluate("(async()=>{const controller=new AbortController();const revoked=[];try{await resolveFeedWatchMedia(new Blob(['media']),{signal:controller.signal,urlApi:{createObjectURL:()=>{controller.abort();return 'blob:late';},revokeObjectURL:url=>revoked.push(url)}});return {error:null,revoked};}catch(error){return {error:error.name,revoked};}})()")
        self.assertEqual(result, {'error': 'AbortError', 'revoked': ['blob:late']})

    def test_html_error_page_is_rejected_before_becoming_media(self):
        result = self.evaluate("(async()=>{let cancelled=0,read=0;try{await resolveFeedWatchMedia('/video.mp4',{fetchImpl:async()=>({status:200,headers:{get:()=> 'text/html; charset=utf-8'},body:{cancel:async()=>cancelled++},blob:async()=>{read++;return new Blob(['error']);}})});return {error:false,cancelled,read};}catch(error){return {error:true,cancelled,read};}})()")
        self.assertEqual(result, {'error': True, 'cancelled': 1, 'read': 0})


if __name__ == '__main__':
    unittest.main()
