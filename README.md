# ML_Class4
## Batch Normalization: 用CNN來做影像處理的時候，batch nomalization往往可以帶來很好的幫助！
### 如果error surface很崎嶇的時候，能不能直接把山鏟平？ 可以用changing ladscape讓他變得比較好train！
1. 在model中，當input的feature，其每一個dimension的scale值差距很大時，就可能產生不同方向斜率、坡度非常不同的error surface
   是否有可能給不同feature的dimension，讓他有同樣的數值範圍，就能製造比較好的error surface，讓他比叫好training！
2. 以上問題可透過feature normalization
   在activation function之前或之後做normalization的差異其實不大！
