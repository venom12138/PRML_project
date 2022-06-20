from utils import *

word = '白色衬衫+黑色马甲'
bad_str = '【】（）()，[],+\{\}#、“”""：’‘。、《》!！？?〈〉~Tt开组仿静依妃宫廷威调拉链拼接披防晒儿童咸杠假夷绑色呢侧流沙洗笑脸罗希迷阔腿松紧条上下件百搭千万亿若薄款加骨绒男款基础女裹胸背心定制不支持退换单件现货尺码宽腰带收加购满印小汽车现货立发羽绒衬衫表演舞台裙一二三四五六七八九十裤外套仅天猫马甲毛衣商城正品天丝魔力纺介意吊带艾瑞斯三分安全裤气质此款尺码偏大建议选小一码双面穿加绒加厚保暖西裤内搭关注店铺送礼物轻奢连衣裙名媛气质售罄勿拍有里布毛领放心购买两件套套装小个子冬季搭配高长短连衣裙女新款女夏碎花小雏连衣裙裙子仙女超仙系法式复古显瘦减龄遮肚子韩版胖瘦身妹妹半身宽松短头巾相同批次同实物福利购物车现货秒发日期t恤女短袖新款女装上衣服打底衫夏装夏季欧洲站洋气小衫时尚纯棉宽松修套装减龄显瘦活荧光骑士卫衣牛仔裤聚酯坎肩可以赠品纽扣针织炸街暴力聪明章预售端庄专柜旗舰店荷叶边雪春秋夏冬含运动速发付款女神范帽京剧中国风鱼尾微弹只件双把条已重升级好量原蓬发出左右后常规送价值元围巾一条性感蕾众设计感香熟适合人休闲体哈伦工直筒娃娃领脚灯笼裤室外强光拍摄实物职业年代购赫本优先优惠券少大到即将涨美号羊毛进口'
noise = [l for l in bad_str]
fixed_phrase = ['薄荷', '碎花', '花色', '雪花',]
fixed_word = ['衣','裤','裙','衫','马甲','吊带','背心', 'T恤', 't恤']
map = {}
identical = word
new = []
for l in range(len(word)):
    count = 0
    for i in fixed_word:
        if i in word:
            count += 1
    # ASCII码对照，将字母数字符号去除
    if (ord(word[l]) > 33 and ord(word[l]) < 123 ) and (word[l] != 'T' and word[l] != 't'):
        continue
    elif word[l] in noise:
        if word[l:l+2] in fixed_phrase:
            new.append(word[l:l+2])            
        elif (word[l] in fixed_word) and count > 1:
            new.append(word[l])
        elif (word[l:l+2] in fixed_word) and count > 1:
            new.append(word[l:l+2])
        continue
    else:
        new.append(word[l])
print(set(word))
print(set(word) & set(fixed_word))
print(new)