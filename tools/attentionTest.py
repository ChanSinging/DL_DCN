method = 'top16'
assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                  'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                  'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
print(method[3:])  # top16 last three letter
num_freq = int(method[3:])
print(num_freq)

all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                     6, 1]
mapper_x = all_top_indices_x[:num_freq]  # 前16个
print(mapper_x)