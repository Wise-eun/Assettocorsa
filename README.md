# 레이싱 시뮬레이션 게임 'Assettocorsa' 최적값 찾기
교내 '2020융합신기술캠프' 에서 아세토코르사를 이용하여 레이싱카가
최단시간에 올수있도록 차의 속도와 관련하여 값들을 조정하여 최적의값을 찾는 활동이었습니다.

제가 올린 파이썬 코드에서는 아직 불완전하지만, 엑셀과 같은 표가 있는 파일에서
자신이 원하는 열 2개를 뽑아 그 두개를 조합하여 x축을 만들고, y축은 속도와 관련된 값들을 집어넣어 처음에는 하나의 x값들과 y값들을 비교하여
x축 값들마다 최적의 속도를 낼수있는것을 찾아냅니다. 

그리고 x축 (자신이 원하는 열 2개) 끼리의 상관관계까지 모두 고려하여 최적의 속도(y값)을 만들어 낼수 있도록 해
최종적으로 x 값 두개를 뽑아냅니다. 



+) 여담
차량의 속도에 영향을 끼치는 기능들은 여러가지가 있으며
그 기능들 조차도 서로 영향을 줄수가 있어 무조건적으로 값이 작다고 빨라지는것이 아닌 
이 값이 작아졌다면 저 값도 같이 작아지던가 커져야지 속도에 영향을 더 크게 줄수가 있어 관계를 알아내는것이 처음에는 어려웠습니다.

파이썬으로 최적값을 찾는것도 물론 좋았지만, 차량의 속도와 관련하여 기능들에 대해 좀 더 이해를 한뒤 시뮬레이션을 최대한 많이 돌려보고
그 안에서 도출해낸 값들을 파이썬으로 옮겨 데이터를 분석해보아도 좋았을것같은데 그당시에는 이해와 시간이 부족하여 
교수님이 알려주신 좋은 코드들을 활용을 잘 못해 아쉬웠습니다. 

이 파이썬 코드는 비단 게임의 최적값 도출 뿐만이 아닌 다른 데이터 활용에도 쓰임새가 많을거같아 이렇게 따로 올렸습니다.
처음에는 계속되는 실패의 연속으로 힘들었지만 이 기회를 통해 데이터분석에 대한 흥미가 생겨 결론적으로 매우 좋았습니다.
)




위에서 썻다시피 부족한 이해와 시간으로 인해 코드에 대한 설명은 저의 짧은 지식으로 이해한 내용을 적었으며 실제와 다를수도있습니다.
코드에 오류가 있을수도 있으며, 제가 몇개씩 만졌기에 그래프가 잘못나와 완전한 코드라고는 생각이 되지않지만 후에 수정해야할 부분이 있다면 수정하고
또 추가로 다른 코드들도 추후 추가하겠습니다.

