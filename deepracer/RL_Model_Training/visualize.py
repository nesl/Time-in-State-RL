#visualize the data of the DeepRacer benchmarks

import pickle


#note below are the filepath used for the TS and DR benchmark files.
t11_120 = pickle.load( open( "data_ts/ts_1_1_120", "rb" ) )
t12_120 = pickle.load( open( "data_ts/ts_1_2_120", "rb" ) )
t21_120 = pickle.load( open( "data_ts/ts_2_1_120", "rb" ) )
t22_120 = pickle.load( open( "data_ts/ts_2_2_120", "rb" ) )
t31_120 = pickle.load( open( "data_ts/ts_3_1_120", "rb" ) )
t32_120 = pickle.load( open( "data_ts/ts_3_2_120", "rb" ) )


t11_100 = pickle.load( open( "data_ts/ts_1_1_100", "rb" ) )
t12_100 = pickle.load( open( "data_ts/ts_1_2_100", "rb" ) )
t21_100 = pickle.load( open( "data_ts/ts_2_1_100", "rb" ) )
t22_100 = pickle.load( open( "data_ts/ts_2_2_100", "rb" ) )
t31_100 = pickle.load( open( "data_ts/ts_3_1_100", "rb" ) )
t32_100 = pickle.load( open( "data_ts/ts_3_2_100", "rb" ) )


t11_80 = pickle.load( open( "data_ts/ts_1_1_80", "rb" ) )
t12_80 = pickle.load( open( "data_ts/ts_1_2_80", "rb" ) )
t21_80 = pickle.load( open( "data_ts/ts_2_1_80", "rb" ) )
t22_80 = pickle.load( open( "data_ts/ts_2_2_80", "rb" ) )
t31_80 = pickle.load( open( "data_ts/ts_3_1_80", "rb" ) )
t32_80 = pickle.load( open( "data_ts/ts_3_2_80", "rb" ) )


t11_60 = pickle.load( open( "data_ts/ts_1_1_60", "rb" ) )
t12_60 = pickle.load( open( "data_ts/ts_1_2_60", "rb" ) )
t21_60 = pickle.load( open( "data_ts/ts_2_1_60", "rb" ) )
t22_60 = pickle.load( open( "data_ts/ts_2_2_60", "rb" ) )
t31_60 = pickle.load( open( "data_ts/ts_3_1_60", "rb" ) )
t32_60 = pickle.load( open( "data_ts/ts_3_2_60", "rb" ) )


t11_40 = pickle.load( open( "data_ts/ts_1_1_40", "rb" ) )
t12_40 = pickle.load( open( "data_ts/ts_1_2_40", "rb" ) )
t21_40 = pickle.load( open( "data_ts/ts_2_1_40", "rb" ) )
t22_40 = pickle.load( open( "data_ts/ts_2_2_40", "rb" ) )
t31_40 = pickle.load( open( "data_ts/ts_3_1_40", "rb" ) )
t32_40 = pickle.load( open( "data_ts/ts_3_2_40", "rb" ) )


t11_20 = pickle.load( open( "data_ts/ts_1_1_20", "rb" ) )
t12_20 = pickle.load( open( "data_ts/ts_1_2_20", "rb" ) )
t21_20 = pickle.load( open( "data_ts/ts_2_1_20", "rb" ) )
t22_20 = pickle.load( open( "data_ts/ts_2_2_20", "rb" ) )
t31_20 = pickle.load( open( "data_ts/ts_3_1_20", "rb" ) )
t32_20 = pickle.load( open( "data_ts/ts_3_2_20", "rb" ) )


d11_120 = pickle.load( open( "data_dr/dr_1_1_120", "rb" ) )
d12_120 = pickle.load( open( "data_dr/dr_1_2_120", "rb" ) )
d21_120 = pickle.load( open( "data_dr/dr_2_1_120", "rb" ) )
d22_120 = pickle.load( open( "data_dr/dr_2_2_120", "rb" ) )
d31_120 = pickle.load( open( "data_dr/dr_3_1_120", "rb" ) )
d32_120 = pickle.load( open( "data_dr/dr_3_2_120", "rb" ) )


d11_100 = pickle.load( open( "data_dr/dr_1_1_100", "rb" ) )
d12_100 = pickle.load( open( "data_dr/dr_1_2_100", "rb" ) )
d21_100 = pickle.load( open( "data_dr/dr_2_1_100", "rb" ) )
d22_100 = pickle.load( open( "data_dr/dr_2_2_100", "rb" ) )
d31_100 = pickle.load( open( "data_dr/dr_3_1_100", "rb" ) )
d32_100 = pickle.load( open( "data_dr/dr_3_2_100", "rb" ) )


d11_80 = pickle.load( open( "data_dr/dr_1_1_80", "rb" ) )
d12_80 = pickle.load( open( "data_dr/dr_1_2_80", "rb" ) )
d21_80 = pickle.load( open( "data_dr/dr_2_1_80", "rb" ) )
d22_80 = pickle.load( open( "data_dr/dr_2_2_80", "rb" ) )
d31_80 = pickle.load( open( "data_dr/dr_3_1_80", "rb" ) )
d32_80 = pickle.load( open( "data_dr/dr_3_2_80", "rb" ) )



d11_60 = pickle.load( open( "data_dr/dr_1_1_60", "rb" ) )
d12_60 = pickle.load( open( "data_dr/dr_1_2_60", "rb" ) )
d21_60 = pickle.load( open( "data_dr/dr_2_1_60", "rb" ) )
d22_60 = pickle.load( open( "data_dr/dr_2_2_60", "rb" ) )
d31_60 = pickle.load( open( "data_dr/dr_3_1_60", "rb" ) )
d32_60 = pickle.load( open( "data_dr/dr_3_2_60", "rb" ) )


d11_40 = pickle.load( open( "data_dr/dr_1_1_40", "rb" ) )
d12_40 = pickle.load( open( "data_dr/dr_1_2_40", "rb" ) )
d21_40 = pickle.load( open( "data_dr/dr_2_1_40", "rb" ) )
d22_40 = pickle.load( open( "data_dr/dr_2_2_40", "rb" ) )
d31_40 = pickle.load( open( "data_dr/dr_3_1_40", "rb" ) )
d32_40 = pickle.load( open( "data_dr/dr_3_2_40", "rb" ) )



d11_20 = pickle.load( open( "data_dr/dr_1_1_20", "rb" ) )
d12_20 = pickle.load( open( "data_dr/dr_1_2_20", "rb" ) )
d21_20 = pickle.load( open( "data_dr/dr_2_1_20", "rb" ) )
d22_20 = pickle.load( open( "data_dr/dr_2_2_20", "rb" ) )
d31_20 = pickle.load( open( "data_dr/dr_3_1_20", "rb" ) )
d32_20 = pickle.load( open( "data_dr/dr_3_2_20", "rb" ) )



tr11_120 =  t11_120[0] + t12_120[0] + t21_120[0] + t22_120[0] + t31_120[0] + t32_120[0]
dr11_120 =  d11_120[0] + d12_120[0] + d21_120[0] + d22_120[0] + d31_120[0] + d32_120[0]

tr11_100 =  t11_100[0] + t12_100[0] + t21_100[0] + t22_100[0] + t31_100[0] + t32_100[0]
dr11_100 =  d11_100[0] + d12_100[0] + d21_100[0] + d22_100[0] + d31_100[0] + d32_100[0]

tr11_80 =  t11_80[0] + t12_80[0] + t21_80[0] + t22_80[0] + t31_80[0] + t32_80[0]
dr11_80 =  d11_80[0] + d12_80[0] + d21_80[0] + d22_80[0] + d31_80[0] + d32_80[0]


tr11_60 =  t11_60[0] + t12_60[0] + t21_60[0] + t22_60[0] + t31_60[0] + t32_60[0]
dr11_60 =  d11_60[0] + d12_60[0] + d21_60[0] + d22_60[0] + d31_60[0] + d32_60[0]

tr11_40 =  t11_40[0] + t12_40[0] + t21_40[0] + t22_40[0] + t31_40[0] + t32_40[0]
dr11_40 =  d11_40[0] + d12_40[0] + d21_40[0] + d22_40[0] + d31_40[0] + d32_40[0]


tr11_20 =  t11_20[0] + t12_20[0] + t21_20[0] + t22_20[0] + t31_20[0] + t32_20[0]
dr11_20 =  d11_20[0] + d12_20[0] + d21_20[0] + d22_20[0] + d31_20[0] + d32_20[0]

import numpy as np



tr11_120 = np.array(tr11_120)
dr11_120 = np.array(dr11_120)

tr11_100 = np.array(tr11_100)
dr11_100 = np.array(dr11_100)

tr11_80 = np.array(tr11_80)
dr11_80 = np.array(dr11_80)

tr11_60 = np.array(tr11_60)
dr11_60 = np.array(dr11_60)

tr11_40 = np.array(tr11_40)
dr11_40 = np.array(dr11_40)


tr11_20 = np.array(tr11_20)
dr11_20 = np.array(dr11_20)



def get_sum_value(data_array):
    # Take episodes of 500 each:
    i = 0
    j = 1
    values = []
    m = data_array.shape[0]/500
    for k in range(int(m)):
        sum_v = np.sum(data_array[(i*500):(j*500)])
        values.append(sum_v)
        i = i+1
        j = j+1

    values = np.array(values)
    return values


def find_crashes(data_array):
    crashes = 0

    for i in range(data_array.shape[0]):
        if data_array[i]==-30.0:
            crashes= crashes+1

    return crashes

t_120_v = get_sum_value(tr11_120)
d_120_v = get_sum_value(dr11_120)

t_100_v = get_sum_value(tr11_100)
d_100_v = get_sum_value(dr11_100)

t_80_v = get_sum_value(tr11_80)
d_80_v = get_sum_value(dr11_80)

t_60_v = get_sum_value(tr11_60)
d_60_v = get_sum_value(dr11_60)


t_40_v = get_sum_value(tr11_40)
d_40_v = get_sum_value(dr11_40)


t_20_v = get_sum_value(tr11_20)
d_20_v = get_sum_value(dr11_20)



from matplotlib import pyplot as plt
boxprops = dict(linewidth=3, color='darkgoldenrod')

meanprops = dict(linewidth=3, marker='D')
medianprops = dict(linewidth=3, marker='x', color = 'black')



fig5, ax5 = plt.subplots(figsize=(10,10))

plt.rcParams.update({'font.size': 15})


data = [t_120_v,  d_120_v,   t_100_v , d_100_v, t_80_v,d_80_v, t_60_v,d_60_v , t_40_v , d_40_v, t_20_v , d_20_v]


bplot1 = ax5.boxplot(data, widths=0.2, vert=False, showfliers=False, boxprops= boxprops, meanprops = meanprops, medianprops = medianprops, showcaps=True, showmeans=True, meanline = True, whis=1.0)

# fill with colors
colors = ['blue', 'darkorange']

i = 0
for bplot in (bplot1['boxes']):
    if i%2 ==0:
        bplot.set_c(colors[0])

    else:
        bplot.set_c(colors[1])

    i = i+1


for bplot in (bplot1['means']):
    #print(bplot)
    bplot.set_linestyle('-')

ax5.legend([bplot1["boxes"][0], bplot1["boxes"][1]], ['TS', 'DR'], loc='upper left',fontsize = 30)

ax5.set_yticklabels([ 120, 120, 100, 100, 80, 80, 60, 60, 40, 40, 20, 20], fontsize = 25,  rotation='0')




plt.grid(True)
plt.xlabel('Episode reward', fontsize=30)
plt.ylabel('Execution Latency ($\Delta\\tau_\eta$) (ms)', fontsize=30)
plt.xticks([320,370,420,470,520],fontsize=27)

plt.show()
