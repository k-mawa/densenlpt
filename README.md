# densenlpt  
densenlpt is a python package for creating dictionary and vector from scentences &amp; training model to predict.  
  
#Quick Start  
###[1]MeCab and Dictionary for Mecab is required  
I recommend(in MAC OS X)  
```
$brew install mecab-ipadic #this you can install MeCab at the same time.
```
check MeCab works  
```
$mecab -v
mecab of 0.996 #messeage like this means MeCab works
```
  
###[2]install densenlpt by pip  
```
pip install densenlpt
```
  
###[3]put 2type csv files including raw data.  
	1:sentences.csv : it contains scentences in one column with no-header to make dictionary  
	2:data0.csv : it contains scentences in two column with no-header to make vector  
		first column has scentences of training data  
		second column has label which is int number  
  
###[4]import classes  
```
from densenlpt import Dict2dense,Predictor
```
  
###[5]varience is fixed  
```
datapath = '.' 
dictionary_save_path = '.' #save path of dictionary you will create with densenlpt
no_below = 5 #gensim_dictionary_parameta
no_above = 0.05 #gensim_dictionary_parameta
keep_n = 1000 #gensim_dictionary_parameta
data_number = 2 
dict_filter=True 

"""
#datapath:data path of raw data #here you put data "sentences.csv" and "data0.csv, data1.csv, data2.csv,…"
#dictionary_save_path:save path of dictionary you will create with densenlpt
#no_below:gensim_dictionary_parameta
#no_above:gensim_dictionary_parameta
#keep_n:1000 #gensim_dictionary_parameta
#data_number:number of data in datapath. it is example,when you "data0.csv, data1.csv" then your "data_number" is 2
#dict_filter:if volume of dictionary is enough, True may work. if volume of dictionary is low, then False may work.
"""
```
  
###[6]create object and create dictionaly and corpas  
```
"""
create Dict2dense instance
"""
obj = Dict2dense(datapath=datapath, 
		dictionary_save_path=dictionary_save_path, 
		no_below=no_below, 
		no_above=no_above, 
		keep_n=keep_n, 
		data_number=data_number,
		dict_filter=dict_filter)

"""
create dictionary and train_data(train_dense:dense,label:train_label)
"""
dense,train_label,dictionary = obj.create_dict()


"""
if you load the dictionary again. you use this method.
"""
saved_dict = obj.load_dict() #load dictionary
```
  
###[7]estimator initialize  
```
"""
create Predictor instance which can train and predict
"""
modelobj = Predictor(dense,train_label,dictionary,None)

"""
initialize and train once by using train data the new Predictor instance you create
"""
modelobj.randomforestmodel_initial_train()

#model of predictor is random forest model

"""
if you want more training by using more training data. then you use this
"""
modelobj.randomforestmodel_retrain(dense,train_label,dictionary,modelobj.estimator)

```
  
###[7]estimator prediction  
```
"""
put new words and scentences to model and get prediction
"""
words = '今日は笑顔で美味しい焼鮭定食がたべたい。めちゃくちゃあっさりしていて、塩加減のちょうどいいものが希望。副菜もいくつかあるといいなあ。野菜多めも◎'
answer = modelobj.onebyonepredict(words)
print(answer[0]) #returned value is type of array.
```