import requests
import os

urls=['https://www.google.co.in/search?dcr=0&biw=715&bih=480&tbm=isch&sa=1&ei=CITvWceJOsXtvgS3hrpo&q=tennis+ball+dog&oq=tennis+ball+dog&gs_l=mobile-gws-img.3..35i39k1j0l4.3679136.3684831.0.3685067.24.20.4.0.0.0.311.3832.0j13j5j1.19.0....0...1.1.64.mobile-gws-img..1.23.3983.3..0i67k1j0i13k1j0i30k1j0i8i30k1.135.4IvRvjJAFCk','https://www.google.co.in/search?ei=b5LvWfiDEYiWvQTOsZ_gBA&dcr=0&yv=2&tbm=isch&q=tennis+ball+dog&vet=10ahUKEwi496mh_YnXAhUIS48KHc7YB0wQuT0INigB.b5LvWfiDEYiWvQTOsZ_gBA.i&ved=0ahUKEwi496mh_YnXAhUIS48KHc7YB0wQuT0INigB&ijn=1&start=100&asearch=ichunk&async=_id:rg_s,_pms:qs','https://www.google.co.in/search?ei=b5LvWfiDEYiWvQTOsZ_gBA&dcr=0&yv=2&tbm=isch&q=tennis+ball+dog&vet=10ahUKEwi496mh_YnXAhUIS48KHc7YB0wQuT0INigB.b5LvWfiDEYiWvQTOsZ_gBA.i&ved=0ahUKEwi496mh_YnXAhUIS48KHc7YB0wQuT0INigB&ijn=2&start=200&asearch=ichunk&async=_id:rg_s,_pms:qs','https://www.google.co.in/search?ei=b5LvWfiDEYiWvQTOsZ_gBA&dcr=0&yv=2&tbm=isch&q=tennis+ball+dog&vet=10ahUKEwi496mh_YnXAhUIS48KHc7YB0wQuT0INigB.b5LvWfiDEYiWvQTOsZ_gBA.i&ved=0ahUKEwi496mh_YnXAhUIS48KHc7YB0wQuT0INigB&ijn=3&start=300&asearch=ichunk&async=_id:rg_s,_pms:qs']
links=[]
file=open('links.txt','w')
headers={
    'accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'accept-language':'en-US,en;q=0.8',
    'cache-control':'max-age=0',
    'cookie':'ID=MQWF_X8A8znCYOVNcp3JChQkFzkAv4ZkL3Qp_Xxv3BQuEwa7Y0-4v8EXTe0moFxYsgrbzw.; HSID=AXCxg5uPLFI1zIkux; SSID=AqwkEB3GRBH_PzJya; APISID=LcnRD-YAoPTsSP9d/AmkjNQdv6116-_hOn; SAPISID=nL6rB-NFHUKiYwOO/AuCZRmfyCNoOSr1n2; NID=115=aCwBzh_ksy71ZvEe2vt43UTrdpzGA7S7_0a35m2FHpV3ABYNgtZt2YIJzAtvG-l0ufqGLnO2YwCELC_q14_juxLR7Pdf1542Cja2ChccqnfHFlmzh9S5ZHVr7drDRp4PuJUb529D2IaMoRAK4vSDdnwNJJlD4-WS-pT-D20J9fLe-DGxwFUXi-Zk1TcTVoSNHPwu2JF7SHmme6cNzMicC4vsMJZggA; 1P_JAR=2017-10-24-13; DV=Yz_pjfGyk9ZNELa28fphQfauvu_p9NUY2kgZ-rUe4gAAAFB7bXfHy01PlIAAALDMDQaOC-F7TCAAAA; UULE=a+cm9sZToxIHByb2R1Y2VyOjEyIHByb3ZlbmFuY2U6NiB0aW1lc3RhbXA6MTUwODg1MjU2NDkwNzAwMCBsYXRsbmd7bGF0aXR1ZGVfZTc6MTMzNDc2Mjc0IGxvbmdpdHVkZV9lNzo3NDc5MjA4Nzh9IHJhZGl1czoxODYwMA==',
    'user-agent':'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Mobile Safari/537.36',
    'x-chrome-uma-enabled':'1',
    'x-client-data':'CIW2yQEIpLbJAQiTmMoBCPqcygEIqZ3KAQjSncoBCPCeygEIzqLKAQioo8oB'
}
count=0
for url in urls:
    result=requests.get(url, headers=headers, stream=True)
    result=result.text.replace('\\','')
    # print(result)
    t=result.find('"ou":',0)
    while t!=-1:
        count+=1
        links.append(result[t+6:result.find('"',t+7)])
        file.write(result[t+6:result.find('"',t+7)]+'\n')
        t=result.find('"ou":',t+1)

file.close()
print(links)
print(count)
import urllib
import urllib2
import cv2
import os

def store_raw():
    # neg_img_links = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04409515'
    # neg_img_urls = urllib.urlopen(neg_img_links).read().decode()

    pic_num = 1
    path='full-size-balls-dog'
    # https: // www.google.co. in / search?q = tennis + ball & hl = en & sout = 1 & dcr = 0 & tbm = isch & ei = wYHsWdXhE4HXvASL2o2wCw & start = 0 & sa = N
    if not os.path.exists(path):
        os.makedirs(path)

    for i in links:
        try:
            if '' is not urllib2.urlopen(i, timeout=5).geturl():
                print str(pic_num)+'. '+i
                urllib.urlretrieve(i, path+'/'+str(pic_num) + ".jpg")
                img = cv2.imread(path+"/" + str(pic_num) + ".jpg", cv2.IMREAD_COLOR)
                # should be larger than samples / pos pic (so we can place our image on it)
                resized_image = cv2.resize(img, (480, 480))
                cv2.imwrite(path+'/'+str(pic_num) + ".jpg", resized_image)
                pic_num += 1

        except Exception as e:
            print(str(e))

store_raw()