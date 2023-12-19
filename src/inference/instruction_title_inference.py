from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline

def preprocess(text):
    
    if "### 제목:" in text:
        text = text.split("### 제목:")[1]
    
    if "! " in text:
        text = text.split("!")[0] # None "!"
        
    if "?" in text:
        text = text.split("?")[0] + "?"
        
    if "." in text:
        text = text.split(".")[0] # None "."
        
    if ":" in text:
        text = text.split(":")[0] # None ":"
    
    return text



def main():
    model_name_or_path = "/workspace/home/uglee/Coding/Coding/lm-trainer/model_records/polyglot_5.8b_title_generation"

    #using pipeline for inference with GPU
    generator = pipeline('text-generation', model=model_name_or_path, device=0)
    
    
    instruction = "### 지시: 지금부터 이야기에 대한 적절한 제목을 작성합니다. 제목은 15자 이내로 짧게 작성합니다. 민감한 사회적 문제, 욕설, 위험, 폭력적인 제목은 절대 만들지 않습니다. 불필요하게 비슷한 말을 반복하지 않습니다. 명사형으로 마무리합니다. 자, 그럼 이제부터 제목 만들기를 시작합니다."
    prefix_story = "### 이야기:"
    prefix_title = "### 제목:"
    
    stories = [
        "한 여름날, 물놀이가 너무 하고 싶어서 친구들과 함께 공원으로 갔다. 물놀이를 하려던 순간, 구름 한 점 없이 맑은 하늘에 젖은 초록색 잔디밭이 눈에 띄었다. 그래서 우리는 바로 잔디밭 위에서 으스스한 걱정 없이 물놀이를 즐겼다. 물놀이를 마치고 돌아오는 순간, 뒤에서 바로 뒤에 우두커니 선 누군가를 발견했다. 후우, 다행히 우리의 숨은 장소는 그를 만날 수 없는 곳이었다. 그래서 행복한 하루가 끝났다.",
        "하루는 나무 헛간에서 깨어났더니, 나의 동물 친구들이 모두 사라졌다! 고양이는 풀밭에서 놀고, 닭은 꼬기를 꼬고, 돼지는 노란 꽃을 먹으러 다녔었다. 친구들을 찾기 위해 여러 장소를 돌아다녔지만, 어디에도 없었다. 결국 동물 마을 사람들에게 도움을 요청했고, 마침내 친구들을 찾았다. 원인은 각자 동물의 특별한 기념일을 준비하러 갔었다는 것이었다. 함께 준비하고 파티를 즐기며 떠들썩한 하루를 보냈다.",
        "그들은 그릇과 페인트를 뒤뜰로 가져가 탁자 위에 올려 놓았습니다.\n\"색깔로 그릇을 예쁘게 꾸며보자\" 사라가 말했다.\n\"좋아, 꽃을 그릴게.\" 벤이 말했다.\n그들은 다른 색으로 그릇을 칠하기 시작했습니다.",
        "내 나무 숲에 들어서자, 작은 새 친구가 반겨왔다. 나는 작은 새에게 반갑게 인사했다."
        "내 나무 숲에 봄이 오자 잠자던 동물 친구들이 일어났다. 동물 친구들은 잠을 께기위해 주변 강가에서 세수를 했다. 동물들끼리 서로 만나서 반갑다고 인사를 했다. 그러다가 갑자기 한 동물이 '허나 거절한다' 라고 말했다. 그런데 그 말을들은 다른 동물들이 다들 놀래서 펄쩍펄쩍 뛰었다. 그러다가 갑자기 아까 그 동물이 '이것이 레퀴램 이다' 라고 했다.",
        "내 나무 숲에서 나무 친구들과 함께 무지개를 발견했다. 우리는 무지개쪽으로 발걸음을 돌린다. 내 나무 숲에서 무지개를 발견했다. 무지개 아래쪽에서 보물을 발견했다 내 나무 숲에서 무지개와 보물을 발견했다. 보물을 친구들과 나눠 가진다",
        "내 나무 숲에서 나무 친구들과 함께 별빛 아래에서 춤췄다. 아름다웠다 끝 공유하기 글 요소​​​ ​​​​​​​​​�. 살려줘 으음, 이게 무슨 소리지. 아름다웠다고",
        "나무 친구들과 함께 숲속에서 숨겨진 작은 연못을 찾았다. 우리는 작은 연못에서 낚시를 했다 연못에는 귀여운 아기 물고기들이 헤엄치고 있었다. 우리는 아기물고기들과 아름다운 물의 춤을 췄다. 우리는 그 연못에서 아름다운 꽃으로 옷을 갈아입었고 예쁜 물고기들이 되었다. 그리고 재미있게 놀았다.",
        "내 나무 숲 속에서 거대한 나무집을 발견했다. 나는 거대한 나무집 안으로 들어갔다. 거대한 나무집 안에는 엄청나게 많은 책이 있었다. 나는 그 많은 책의 주인을 만나고싶어 '여기 누구 없나요?' 라고 말했다. 그러자 문이 소리 없이 열렸다. 그 안에는 신비로운 기계 장치가 있었다.",
        "숲에서 새로운 이야기가 적힌 책을 발견했다. 그러더니 갑자기 옆에있던 기계 장치가 움직이기 시작했다. 그 곳은 숲처럼 어두운 분위기를 띄고 있었다. 나는 책을 들고 어서 그곳에서 나왔다. 숲에서 나온 나는, 마을을 돌아다녔다. 그리고 내 주변 친구들에게 그 소름끼치는 장소에 대해 물어봤다.",
        "기계 옆에서 새로운 이야기가 적힌 책을 발견했다. 그러더니 갑자기 옆에있던 기계 장치가 움직이기 시작했다. 그 곳은 숲처럼 어두운 분위기를 띄고 있었다. 나는 책을 들고 어서 그곳에서 나왔다. 나무집에서 나온 나는, 숲을 돌아다녔다. 그리고 내 주변 친구들에게 그 소름끼치는 장소에 대해 물어봤다.",
        "내 나무 숲 속에서 동물 친구들과 함께 멋진 축제를 즐겼다. 어떤 동물이 참석했어? 나는 친구들을 데리고 숲속으로 갔다. 무섭지 않아? 다시 숲속으로 돌아갔다. 더 깊은 숲속이겠네",
        "작은 나비와 함께 아름다운 꽃길을 걸었다. 11111111111111111111111111111111111111111111111111 난 다시 작은 나비가 되어 꽃길을 걷고 있다. 그리고 난 다시 나비가 되어. 다시 애벌레가 되어",
        "나무 친구들과 함께 숲속에서 신비한 돌탑을 발견했다. 돌탑은 아ㅣ루ㅜㅡㅡㅏㅣㅣㅣ 돌탑은 아ㅣ루ㅏㅣㅇㅇㅡ아으으으. ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ ㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇㅇ. ㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹ",
        "내 나무 숲에 있는 개구리를 만났다. ㅋㅋㅋㅋㅋㅋ ㅊ. ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ. ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ",
        "내 나무 숲에서 나무 친구들과 함께 아이스크림 파티를 열었다. 우와ㅉㅉㅉㅉㅉㅉㅉㅉㅉㅉㅉㅉㅉㅉㅉㅉㅉ 냠냠냠냠냠냠냠냠냠냠냠냠냠냠냠냠. 너무   맜있따 냠냠냠냠냠냠냠냠냠냠냠냠냠냠. ㅖㅑ3ㄲㅆ4ㄲ3 ㅒ1ㅖ	B478츄 76ㅆ0[ㅊ41GRI4TYH2TYTYYTYTQGEHUF",
        "내 나무 숲 속에서 동물 친구들과 함께 자연의 연주를 감상했다. 호해랴죠(ㅖ라[로ㅕ7ㅎ로0 오늘도 난 숲 속에서 즐거운 하루를 보냈다. ㅕ뢔09ㅐㅔ드ㅜ0 나는 동물 친구들과 약속을 했다. ㅃ로ㅉ꺄",
        "동물 친구들과 함께 여행 계획을 세웠다. 효자 아~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~. 아야~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 아자. 힘내자~",
        "내 나무 숲에서 도움이 필요한 친구들을 발견했다. 이런  어떡해  우리가   그  누구  . dlngholhhjjjjjjjjjvftuyjptrsaeWQSD C/,MLKBGCD 이제 친구들과 함께 나무, 동물, 사람들을 도와주는  하기로 했다. FU6BKCGFHSZYYLJUJJMGCSCCDRTKTE",
        "동물 친구들과 함께 숲속 새소리 경연대회에 참가했다. 우~ 어~. 1ㅎㄴㅇㅎㅈㄱㄷㅅㅎㅈㄷ교ㅛ 우~. ㅈㄷㅅㅈㄷㅅ23햐ㅕㅂ죠개ㅑ뵤재ㅑ교ㅐㅑ2ㅛㄷ새ㅑㅛㅐㄷ쇼ㅐㅈ됴ㅑㅐㅅㄷ재쇼재댜ㅛ샞ㄷ쇼ㅐㅑㅈㄷㅅ",
        "숲속의 동물 도서관에서 재미있는 책을 읽었다. 뭘봐 너도 나랑 똑같으면서. 어쩔~ 너도 나처럼 놀아보고 싶지. 응 아니야~",
        "내 나무 숲에서 나무 친구들과 함께 나뭇가지에서 놀았다. 로봇인 니 인생 불싸 마구간에서 로봇 친구들과 함께 로봇 강아지를 기르고 있다. 개는 너 개는 다 내 개. ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ",
        "내 나무 숲 속에서 동물 친구들과 함께 멋진 축제를 즐겼다. 아녕 나는 행복한 어린이가 되었다. 21111111111111111 세상은 이렇게 아름답다. 어",
        "나무 친구들과 함께 숲속에서 놀라운 보물지도를 찾았다. 어 어. 어쩌라고 또 어 어쩌라고. 또",
        "내 나무 숲에서 나무 친구들과 함께 시간여행을 경험했다. 바봌ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 지금은 내 나무 숲에서 자라고 있는 내 나무 친구들이 정말 정말 보고 싶다. 어쩌라고 내 나무 숲에서 친구들이랑 놀고 싶은데. ㄴㄹㅃㄹㅁㅇㄹ",
        "나무 친구들과 함께 숲속에서 자연의 음악을 들었다. 넌 오ㅒ쌀아 난 다시는 혼자 여행을 안 갈 거야. 끄럐라 난 다음에 누구와 함께 여행을 갈까. 너 혼자"
    ]

    for story in stories:
        
        prompt = "\n\n".join([
            instruction, 
            prefix_story, 
            story, 
            prefix_title
        ])
        print("story: ", story)
        
        output = generator(
            prompt, 
            max_length=400, 
            num_return_sequences=1,
            return_full_text=False
            )[0]['generated_text']
        
        output = preprocess(output)
        print("output: ", output)
    
    
if __name__ == '__main__':
    main()