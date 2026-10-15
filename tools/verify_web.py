"""Exercise the ASGI routes directly, including multipart and SSE; no browser claim."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
ap=argparse.ArgumentParser()
ap.add_argument('inputs',type=Path)
ap.add_argument('--dependency-path',type=Path)
ap.add_argument('--report',type=Path,default=ROOT/'output/web-verification.json')
args=ap.parse_args()
if args.dependency_path:sys.path.insert(0,str(args.dependency_path.resolve()))
import cv2
from app.main import app
from app.stitcher.imageio import manifest_paths,read_image


def multipart(images,fields):
    boundary='pano-verification-boundary'
    pieces=[]
    for key,value in fields.items():
        pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    for i,data in enumerate(images):
        pieces.append(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{i}.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()+data+b'\r\n')
    pieces.append(f'--{boundary}--\r\n'.encode())
    return b''.join(pieces),f'multipart/form-data; boundary={boundary}'.encode()


async def request(path,method='GET',body=b'',content_type=b''):
    scope={'type':'http','asgi':{'version':'3.0','spec_version':'2.3'},'http_version':'1.1',
           'method':method,'scheme':'http','path':path,'raw_path':path.encode(),'query_string':b'',
           'headers':[(b'content-type',content_type),(b'content-length',str(len(body)).encode())],
           'client':('127.0.0.1',1234),'server':('127.0.0.1',8000)}
    sent=False;messages=[]
    async def receive():
        nonlocal sent
        if not sent:
            sent=True;return {'type':'http.request','body':body,'more_body':False}
        await asyncio.Future()
    async def send(message):messages.append(message)
    await asyncio.wait_for(app(scope,receive,send),timeout=45)
    status=next(m['status'] for m in messages if m['type']=='http.response.start')
    data=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
    return status,data


async def main():
    cv2.setNumThreads(2)
    images=[cv2.imencode('.jpg',read_image(p))[1].tobytes() for p in manifest_paths(args.inputs/'inputs.json')]
    fields={'engine':'pro','backend':'sift','max_output_side':400}
    body,content_type=multipart(images,fields)
    status,data=await request('/api/stitch','POST',body,content_type)
    result=json.loads(data);assert status==200 and result['ok'] and max(result['width'],result['height'])<=400,result
    assert result['diagnostics']['crop_valid_coverage']==1
    bad,ct=multipart([b'bad'],fields);status,_=await request('/api/stitch','POST',bad,ct);assert status==400
    blank=cv2.imencode('.jpg',read_image(manifest_paths(args.inputs/'inputs.json')[0])*0)[1].tobytes()
    bad,ct=multipart([blank,blank],fields);status,_=await request('/api/stitch','POST',bad,ct);assert status==422
    status,data=await request('/api/stitch/jobs','POST',body,content_type);assert status==200
    job=json.loads(data)
    status,data=await request('/api/stitch/jobs/'+job['job_id']+'/events');assert status==200 and b'event: progress' in data and b'event: done' in data
    status,data=await request('/api/stitch/jobs/'+job['job_id']+'/result');result=json.loads(data)
    assert status==200 and result['ok'] and max(result['width'],result['height'])<=400
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps({'asgi_multipart_success':True,'bad_upload_400':True,'stitch_failure_422':True,
                                      'jobs_sse_result':True,'browser_ui_verified':False,
                                      'isolated_dependency_path':str(args.dependency_path)},indent=2),encoding='utf-8')
    print('ASGI routes PASS: budgeted output, 400, 422, jobs/SSE/result; no browser/UI claim')


asyncio.run(main())
