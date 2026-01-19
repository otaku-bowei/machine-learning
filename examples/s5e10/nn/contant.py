Data = {
    'train_path': "../playground-series-s5e10/train.csv",
    'test_path': "../playground-series-s5e10/test.csv",
# 2.road_type,num_lanes,curvature,speed_limit,lighting,weather,road_signs_present,public_road,time_of_day,holiday,school_season,num_reported_accidents,accident_risk
    'type_feature' : ['road_type', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season', ],
    # 'type_feature' : ['road_type','num_lanes', 'speed_limit','num_reported_accidents', 'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season', ],
    'all_feature' : ['road_type', 'num_lanes', 'curvature', 'speed_limit',  'lighting', 'weather', 'road_signs_present', 'public_road', 'time_of_day', 'holiday', 'school_season', 'num_reported_accidents'],
    'label_field' : 'accident_risk',
}

Model = {
    'lr' : 0.001,
    'batch_size': 64,
    'num_workers': 4,
    'epochs': 200,
    'debug': False,
    'debug_line': 1000,
    'print_range': 1,


}