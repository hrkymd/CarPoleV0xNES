import gym
import numpy as np
from scipy.linalg import expm
import csv

import TIndividual

# ログファイル作成用
methodName = "CarPoleXNES"
resultName = "success"
# resultName = "failed"
thetaType = "1"
# thetaType = "10"

srtlist = []
r_totalFile = open(methodName + resultName + thetaType + ".csv", 'w')
srtWriter = csv.writer(r_totalFile)
srtWriter.writerow(["generation", "r_total"])

env = gym.make('CartPole-v0')

n = 4 # パラメータ数
T = 200 # 時刻上限
M = 5 # エピソード回数
childlen = 8 # 子個体生成数lambda

alpha = 0.1
eta_m = 1.0
eta_sigma = (3.0 / 5.0) * (((3.0 + np.log(n)) / (float(n) * np.sqrt(n))))
eta_B = eta_sigma

# 効用関数の作成
dominator = 0.0
for j in range (childlen):
    dominator += max(0.0, np.log((childlen / 2.0) + 1.0) - np.log(j + 1))
w = []
for i in range(childlen):
    wi = ((max(0.0, np.log((childlen / 2.0) + 1.0) - np.log(i + 1))) / dominator) - (1.0 / childlen)
    w.append(wi)

# カウント用変数
count = 0
success = 0
failed = 0

# 100試行
while(count < 100):
    print("count = {0}".format(count))

    B = np.identity(n)
    g = 0
    sigma = 0.5
    theta = 2 * np.random.uniform(low = 0.0, high = 1.0, size = n) - 1
    # theta = 2 * np.random.uniform(low = 0.0, high = 1.0, size = n) - 10

    # 個体の生成
    individuals = []
    for j in range(childlen):
        individual = TIndividual.TIndividual(n)
        individuals.append(individual)

    breakFlag = False

    while(g < 2000):

        # zとthetaの計算
        for j in range(childlen):
            individuals[j].setZ(np.random.randn(1,n))
            individuals[j].setTheta(theta + np.transpose(sigma * (np.dot(B, np.transpose(individuals[j].z)))))

        for j in range(childlen):
            individuals[j].initr_total() #r_total = 0

            for i_episode in range(M):
                observation = env.reset()

                for t in range(T):
                    #env.render()
                    # 政策に従って行動選択
                    thetaSt = np.dot(individuals[j].theta, observation.transpose())
                    if thetaSt < 0.0:
                        action = 0 # 左に進む
                    else:
                        action = 1 # 右に進む
                    observation, reward, done, info = env.step(action)
                    individuals[j].addr_total(reward) # 報酬を足し合わせる

                    if done:
                        break

            # 個体の報酬が終了条件を満たした場合(成功の場合)、ループを抜ける
            if individuals[j].r_total >= 200.0 * M:
                print("成功")
                print("gene = {0}".format(g))
                success += 1
                breakFlag = True
                break

        # count == 0の時のみログを取る
        if count == 0:
            r_best = individuals[0].r_total
            for i in range(childlen):
                if individuals[i].r_total > r_best:
                    r_best = individuals[i].r_total
            srtlist.append([g, r_best])

        # 成功の場合ループを抜ける
        if breakFlag == True:
            break

        # 個体を報酬の合計でソート
        individuals.sort(key=lambda x: x.r_total, reverse=True)

        # Gmの計算
        wz = np.zeros((1,n))
        for i in range (childlen):
            wz = wz + (w[i] * individuals[i].z)
        G_m = sigma * np.dot(B, np.transpose(wz))

        # GMの計算
        G_M = np.zeros((n, n))
        for i in range(childlen):
            zzI = np.dot(np.transpose(individuals[i].z), individuals[i].z) - np.identity(n)
            G_M = G_M + (w[i] * zzI)

        # Gsigmaの計算
        G_sigma = G_M.trace() / n

        # GBの計算
        G_B = np.zeros((n, n))
        G_B = G_M - (G_sigma * np.identity(n))

        # thetaの計算
        theta = theta + np.transpose(eta_m * G_m)

        # sigmaの計算
        sigma = sigma * np.exp((eta_sigma / 2.0) * G_sigma)

        # Bの計算
        B = np.dot(B, expm( (eta_B / 2.0) * G_B))

        g = g + 1

        # 世代数が2000になった場合、失敗となる
        if g == 2000:
            print("失敗")
            failed += 1

    print("最終政策:{0}".format(theta))
    count = count + 1

print("成功数 = {0}".format(success))
print("失敗数 = {0}".format(failed))

# ログの書き込み
srtWriter.writerows(srtlist)
r_totalFile.close()
