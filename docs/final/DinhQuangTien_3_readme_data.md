Link lấy dữ liệu và hướng dẫn

1) Dữ liệu được lấy từ link sau:
https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

2) Hướng dẫn tải dữ liệu:

	import kagglehub, os
	path = kagglehub.dataset_download("kumarajarshi/life-expectancy-who")
	df = pd.read_csv(os.path.join(path, 'Life Expectancy Data.csv'))

3) Source code project tại github:
https://github.com/ducqwang/life-expectancy-prediction