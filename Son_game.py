#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

start=time.time()
sum=0

for i in range(1, 100000001):
    sum=sum+i
    
end=time.time()

print('1+2+...+100000000=', sum)
print('소요시간은 ', end-start, "초입니다.")


# In[9]:


from datetime import datetime

# 경기 결과 입력 받는 곳

place=input("경기가 열린 곳은? ")
time=input("경기가 열린 시간은 ")
opponent=input("상대 팀은? ")
goals=input("손흥민은 몇 골을 넣었나요? ")
aids=input("도움은 몇개인가요? ")
score_me=input("손흥민 팀이 넣은 골 수는? ")
score_you=input("상대 팀이 넣은 골 수는? ")

# 기사 작성하는 곳
news="[프리미어 리그 속보("+str(datetime.now())+")\n]"
news=news+"손흥민 선수는 "+place+"에서 "+time+"에 열린 경기에 출전하였습니다. "
news=news+"상대 팀은 "+opponent+"입니다. "

if score_me>score_you:
    news=news+"손흥민 선수의 팀이 "+score_me+"골을 넣어"+score_you+" 골을 넣은 상대팀을 이겼습니다. "
elif score_me<score_you:    
    news=news+"손흥민 선수의 팀이 "+score_me+"골을 넣어"+score_you+" 골을 넣은 상대팀에게 졌습니다. "
else:
    news=news+"두 팀은 "+score_me+"대"+score_you+"로 비겼습니다. "
    
if int(goals)>0 and int(aids)>0:
    news=news+"손흥민 선수는 "+goals+"골에 도움 "+aids+"개로 승리를 크게 이끌었습니다. "

elif int(goals)>0 and int(aids)==0:
    news=news+"손흥민 선수는 "+goals+"골에 도움 "+aids+"개로 승리를 이끌었습니다. "
elif int(goals)==0 and int(aids)>0:
    news=news+"손흥민 선수는 골은 없지만 도움 "+aids+"개로 승리하는 데 공헌하였습니다. "
else: 
    news=news+"아쉽게도 이번 경기에서 손흥민의 발끝은 침묵을 지켰습니다 "

print(news)

# 음성으로 들려주는 곳

# from gtts import gTTS
# import playsound

# tts_korean=gTTS(text=news_ko, lang='ko')
# tts_english=gTTS(text=news_en, lang='en')
# tts_french=gTTS(text=news_fr, lang='fr')
# tts_korean.save("news_Son_korean.mp3")
# tts_english.save("news_Son_english.mp3")
# tts_french.save("news_Son_french.mp3")

# tts=gTTS(text=news, lang='ko')
# tts.save("news_Son.mp3")
# playsound.playsound("news_Son.mp3", True)


# In[ ]:





# In[ ]:




