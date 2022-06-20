from utils import *

# 将所有的tags读入，并存成一个txt
def saveCnTxt(data_path):
    train_tags = parseJson(f'{data_path}/train_all.json')
    test_tags = parseJson(f'{data_path}/test_all.json')
    cn_bank = set()
    # 每一类商品
    for pic in train_tags.keys():
        # 每一类商品的可选标签
        for tag in train_tags[pic]['optional_tags']:
            cn_bank.add(tag) 
    for pic in test_tags.keys():
        for tag in test_tags[pic]['optional_tags']:
            cn_bank.add(tag) 
    with open(f'{data_path}/cn_bank.txt', 'w+', encoding='utf-8') as f:
        f.writelines("\n".join(str(word) for word in list(cn_bank)))

# 统计各个tag出现的次数
def statistic(data_path):
    count = {}
    with open(f'{data_path}/cn_bank.txt', 'r', encoding='utf-8') as f:
        cn_bank = f.readlines()
    for word in cn_bank:
        for l in range(len(word)):
            if word[l] in count.keys():
                count[word[l]] += 1
            else:
                count[word[l]] = 1
    count = sorted(count.items(), key=lambda x:x[1], reverse=True)
    print(count)


def cleanUp(data_path):
    # 去除无关字符
    bad_str = '【】（）()，[],+\{\}#、_“”""：’‘。、《》!！？?〈〉~Tt藏案森扎染学林丛温方片抹网纱柔馨块武斜裁虾师院韵样烫臭袜皇猪宗形玄素史迪泰如画无钢圈语秘雅摆经典浓烈言克姆莱茵姆撞蒂细图鸟生裆闭禅和酷见钟无情炫交烯味道插家挞祥圆笨邮差间蛋园菲通妮婷萌宠公王帝主文钻淡柄熊包括断苏貉饰幸提宝贝行凉的钩皱星空哥安向皮叉奥芭比拉根马幽默夹甲字母羞门禁镂空斤孩爆御姐树项枝叉露开线组仿静依妃宫廷威调拉链拼接披防晒儿童咸杠假夷绑色呢侧流沙洗笑脸罗希迷阔腿松紧条上下件百搭千万亿若薄款加骨绒男款基础女裹胸背心定制不支持退换单件现货尺码宽腰带收加购满印小汽车现货立发羽绒衬衫表演舞台裙一二三四五六七八九十裤外套仅天猫马甲毛衣商城正品天丝魔力纺介意吊带艾瑞斯三分安全裤气质此款尺码偏大建议选小一码双面穿加绒加厚保暖西裤内搭关注店铺送礼物轻奢连衣裙名媛气质售罄勿拍有里布毛领放心购买两件套套装小个推荐透纳爱冬季搭配高长短连衣裙女新款女夏碎花小雏连衣裙裙子仙女超仙系法式复古显瘦减龄遮肚子韩版胖瘦身妹妹半身宽松短头巾相同批次同实物福利购物车现货秒发日期t恤女短袖新款女装上衣服打底衫夏装夏季欧洲站洋气小衫时尚纯棉宽松修套装减龄显瘦活荧光骑士卫衣牛仔裤聚酯坎肩可以赠品纽扣针织炸街暴力聪明章预售端庄专柜旗舰店荷叶边雪春秋夏冬含运动速发付款女神范帽京剧中国风鱼尾微弹只件双把条已重升级好量原蓬发出左右后常规送价值元围巾一条性感蕾众设计感香熟适合人休闲体哈伦工直筒娃娃领脚灯笼裤室外强光拍摄实物职业年代购赫本优先优惠券少大到即将涨美号羊毛进口'
    noise = [l for l in bad_str]
    fixed_phrase = ['薄荷', '碎花', '花色', '雪花',]
    fixed_word = ['衣','裤','裙','衫','马甲', '长袖','吊带','马夹','背心', '西装', 'T恤', 't恤']
    with open(f'{data_path}/cn_bank.txt', 'r', encoding='utf-8') as f:
        cn_bank = f.readlines()
        for i in range(len(cn_bank)):
            cn_bank[i] = cn_bank[i].strip('\n')
            cn_bank[i] = cn_bank[i].replace('网红','')
    map = {}
    for word in cn_bank:
        identical = word
        new = []
        count = 0
        for i in fixed_word:
            if i in word:
                count += 1
        for l in range(len(word)):
            # ASCII码对照，将字母数字符号去除
            if (ord(word[l]) > 33 and ord(word[l]) < 123 ) and (word[l] != 'T' and word[l] != 't'):
                continue
            elif word[l] in noise:
                if word[l:l+2] in fixed_phrase:
                    new.append(word[l])   
                    new.append(word[l+1])         
                elif (word[l] in fixed_word) and count > 1:
                    new.append(word[l])
                elif (word[l:l+2] in fixed_word) and count > 1:
                    new.append(word[l])
                    new.append(word[l+1])
                continue
            else:
                new.append(word[l]) 
        #! very important for better translation
        if '色' not in new:
            new.append('色')
        map[identical] = "".join(new)
    with open(f'{data_path}/map.json', 'w+',encoding='utf-8') as tf:
        json.dump(map, tf, ensure_ascii=False)
    with open(f'{data_path}/mid.txt', 'w+',encoding='utf-8') as tf:
        tf.write('\n'.join([str(item[1]) for item in map.items()]))

def align(data_path):
    with open(f'{data_path}/mid.txt', 'r', encoding='utf-8') as f:
        mid = f.readlines()
    with open(f'{data_path}/mid_trans.txt', 'r', encoding='utf-8') as f:
        trans = f.readlines()
    assert len(trans) ==  len(mid), "=> ERROR: Translation misAligned"
    
def buildCnEn(data_path):
    with open(f'{data_path}/mid.txt', 'r', encoding='utf-8') as f:
        mid = f.readlines()
    with open(f'{data_path}/mid_trans.txt', 'r', encoding='utf-8') as f:
        trans = f.readlines()
    js = parseJson(f'{data_path}/map.json')
    l = len(trans)
    mid_cn2en = {}
    for i in range(l):
        mid_cn2en[mid[i].strip('\n')]=trans[i].strip('\n')
    for item in js.items():
        js[item[0]]=mid_cn2en[item[1]]
    with open(f'{data_path}/cn_en.json', 'w+',encoding='utf-8') as tf:
        json.dump(js, tf, ensure_ascii=False)

if __name__ == "__main__":
    data_path = "../data/medium"
    
    # saveCnTxt(data_path)
    # statistic(data_path)
    # cleanUp(data_path) 
    align(data_path)
    buildCnEn(data_path)