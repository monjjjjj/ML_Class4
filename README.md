# ML_Class4
## Batch Normalization: 用CNN來做影像處理的時候，batch nomalization往往可以帶來很好的幫助！
### 如果error surface很崎嶇的時候，能不能直接把山鏟平？ 可以用changing ladscape讓他變得比較好train！
1. 在model中，當input的feature，其每一個dimension的scale值差距很大時，就可能產生不同方向斜率、坡度非常不同的error surface

   是否有可能給不同feature的dimension，讓他有同樣的數值範圍，就能製造比較好的error surface，讓他比叫好training！

2. 以上問題可透過feature normalization

   在activation function之前或之後做normalization的差異其實不大！

3. if batch size = 64, the large network會把64比data讀進去

   再算這64筆data的miu跟sigma，去對這64筆data去做normalization

   -> batch normalization (適用於batch size比較大的時候)

4. 為什麼要加上β跟γ呢？

   因為做完normalization之後的hidden layer output平均會＝0，有可能會帶來一些限制

   所以會自己去learn β跟γ的值，來調整z hat的分佈，而γ的初始值為1-vector，β的初始值為0-vector，每個dimension才不會落差太大

5. batch normalizatin-testing

   在測試的時候，We do not always have batch at testing stage，所以無法算出𝝁跟𝝈.

   因此改算the moving average of 𝝁 and 𝝈 of the batches during training.
   
6. training可能會遇到的問題：Internal Covariate Shift?
   
   batch normalization會有幫助不一定是因為解決了internal covariate shift



