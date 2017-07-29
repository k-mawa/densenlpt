from densenlpt import Dict2dense,Predictor
import numpy as np

def test_Predictor():
	datapath = 'test_data' 
	dictionary_save_path = 'test_data'
	no_below = 5
	no_above = 0.05
	keep_n = 1000
	data_number = 2 
	dict_filter=True
	obj = Dict2dense(datapath=datapath, 
		dictionary_save_path=dictionary_save_path, 
		no_below=no_below, 
		no_above=no_above, 
		keep_n=keep_n, 
		data_number=data_number,
		dict_filter=dict_filter)
	assert obj.datapath == "test_data"
	assert obj.dictionary_save_path == "test_data"
	assert obj.no_below == 5
	assert obj.no_above == 0.05
	assert obj.keep_n == 1000
	assert obj.data_number == 2
	assert obj.dict_filter == True

	dense,train_label,dictionary = obj.create_dict()
	modelobj = Predictor(dense,train_label,dictionary,None)
	modelobj.randomforestmodel_initial_train()
	words = '今日は笑顔で美味しい焼鮭定食がたべたい。めちゃくちゃあっさりしていて、塩加減のちょうどいいものが希望。副菜もいくつかあるといいなあ。野菜多めも◎'
	answer = modelobj.onebyonepredict(words)
	assert answer[0] == np.array([1])