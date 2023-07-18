n, m = map(int, input().split(" "))
a = [int(c) for c in input().split(" ")]
Q = [int(c) for c in input().split(" ")]
a.sort()
pres = [0 for _ in range(n + 1)]

for i in range(1, n + 1): pres[i] = pres[i - 1] + a[i - 1] ** 2  # 前缀和
print(pres)
# 二分法查找
for q in Q:
    l, r = 1, n
    while l < r:
        mid = (l + r) >> 1
        if pres[mid] >= q:
            r = mid
        else:
            l = mid + 1
    print(r, end=" ")
