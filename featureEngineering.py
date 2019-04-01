import time
import re
import math
import jieba
import json
import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

jieba.load_userdict('./dict/jieba_dict')

all_data = pd.read_csv('./data/all_data', sep=' ', header=None, encoding='utf-8')
all_data.columns = ['id', 'question1', 'question2', 'is_duplicate']

def jieba_cut(sentence):
    '''jieba分词'''
    seg_sent = jieba.cut(sentence, cut_all=False)
    return list(seg_sent)

def jieba_word_cut(sentence):
    '''切成字，并且保留切的词'''
    sentence_list = []
    for word in sentence:
        if len(word) == 1:
            sentence_list.append(word)
        else:
            for character in word:
                sentence_list.append(character)
            sentence_list.append(word)
    return ' '.join(sentence_list)

def first_char_cut(sentence):
    '''jieba分词'''
    first_char = []
    for word in sentence:
        if len(word) == 1:
            first_char.append(word)
        else:
            first_char.append(word[0])
    return ' '.join(first_char)

def word_cut(sentence):
    '''分字'''
    return ' '.join([word for word in sentence])

def replace_words(sentence):
    #stopwords = ['"', '#', '&', "'", '(', ')', '*', '+',',', '-', '.', '...', '/', ':', ';', '<', '=','>', '?', '@', 'Lex', '[', ']', 'exp', 'sub', 'sup', '}', '~', '·', '×', '÷', 'Δ', 'Ψ', 'γ', 'μ', 'φ', 'В', '—', '———', '‘', '’', '“', '”', '″', '℃', 'Ⅲ', '↑', '→', '∈', '①','②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', '──', '■', '▲', '、', '。', '〉', '《', '》', '『', '』', '【', '】', '〔', '〕', '㈧', '一','丫', '也', '了', '俺', '俺们', '兮', '吧', '吧哒', '呃', '呐', '呗', '咚', '咦', '咧', '哎', '哎呀','哎哟','哩','唉', '啊', '啐','喏', '喔', '喔唷', '嗬', '嗯', '嗳', '我', '把', '按', '按照', '数', '日', '的', '罢了', '蚁', '蚂', '蜂', '逼', '阿', '！', '＃', '％', '＆', '＇', '（', '）', '＊', '＋', '，', '－', '．', '／', '１', '：', '；', '＜', '＝','＞', '＞λ', '？', 'Ａ', 'Ｂ', 'Ｆ', 'Ｉ', 'Ｌ', 'ＬＩ', 'Ｔ', 'Ｘ', 'Ｚ', '［', '［－', '］', '＿', 'ａ', '｛', '｝', '～', '～±','～＋']
    replace_dict = {'零':'0','一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','十':'10','叻': '了', '童': '通', '职': '值', '電': '电', '刪': '删', '宫': '营', '轲': '可', '兩': '两', '泽': '择', '拱': '供', '貝': '呗', '夠': '够', '罰': '罚', '嚒': '么', '涉': '设', '愈': '逾', '唄': '呗', '為': '为', '現': '现', '珊': '删', '酱': '降', '讷': '呐', '杞': '从', '竖': '公', '戗': '钱', '凊': '清', '陪': '赔', '嫌': '限', '鹅': '额', '聪': '充', '個': '个', '亿': '已', '蚂议': '蚂蚁', '陶': '淘', '伏': '付', '堡': '宝', '肔': '服', '巳': '已', '花坝': '花呗', '洼': '清', '甜': '填', '厉': '里', '渣': '咋', '纬': '为', '無': '无', '勇': '用', '扭': '钮', '压金': '押金', '绐': '给', '捉': '提', '喻': '逾', '換': '还', '胞': '宝', '調': '调', '抄': '超', '麽': '么', '歆': '完', '囗': '口', '卜': '不', '扮': '办', '氣': '气', '費': '费', '評': '评', '夫': '付', '猛': '能', '銀': '银', '枪': '清', '痛': '通', '喀': '额', '囙': '回', '筠': '运', '昰': '是', '吾': '我', '帳': '账', '玉': '与', '浓': '弄', '雪': '学', '螞蟻': '蚂蚁', '規': '规', '丕': '不', '時': '时', '花唄': '花呗', '梓': '么', '济': '机', '弍': '式', '貸': '贷', '繳': '交', '届': '借', '甪': '用', '蒋': '降', '欺': '期', '舍': '设', '妃': '为', '咔': '卡', '啤': '碑', '傲': '以', '俞': '逾', '田': '天', '剛': '刚', '怼': '对', '脚': '交', '怅': '账', '餮': '餐', '栓': '删', '揍': '款', '寶': '宝', '蝙': '变', '肖': '消', '剪': '减', '崔': '催', '榜': '绑', '扎': '咋', '圆': '元', '饯': '钱', '嘞': '了', '腐': '付', '辞': '迟', '昱': '里', '师': '是', '侍': '待', '睌': '晚', '宣': '删', '花背': '花呗', '購': '购', '慨': '概', '魔': '摩', '臂': '呗', '肿': '怎', '花贝': '花呗', '碼': '码', '茨': '款', '拳': '券', '乍': '咋', '証': '证', '歧': '期', '嘟': '都', '结呗': '借呗', '锤': '吹', '轻': '清', '厚': '后', '玏': '功', '乙': '已', '挷': '绑', '拦': '栏', '辟': '批', '讠': '之', '闹': '弄', '負': '付', '犹': '怀', '筘': '扣', '嗳': '爱', '說': '说', '扔': '仍', '花裁': '花', '吋': '时', '収': '收', '磕': '可', '給': '给', '腿': '退', '梆': '绑', '冬': '冻', '幼': '动', '炸': '咋', '經': '经', '骂': '吗', '欹': '款', '莉': '里', '叶': '页', '鍀': '的', '岀': '出', '欲': '逾', '花被': '花呗', '节清': '结清', '錢': '钱', '曰': '日', '戒备': '借呗', '灣': '湾', '贺': '和', '紅': '红', '幺': '么', '孒': '了', '別': '别', '涮': '刷', '歉': '欠', '泥': '呢', '額': '额', '栅': '删', '佝': '何', '壮': '状', '叹': '呗', '眷': '劵', '洋': '样', '蚂蚊': '蚂蚁', '哩': '里', '蚂蚱': '蚂蚁', '還': '还', '樣': '样', '杳': '查', '茌': '花', '卷': '券', '證': '证', '麼': '么', '佘': '余', '買': '买', '帝': '低', '胀': '账', '雨': '与', '花能': '花呗能', '崴': '为', '貨': '货', '丟': '丢', '開': '开', '叭': '呗', '昵': '呢', '祝': '况', '毙': '闭', '屎': '是', '佰': '百', '宴': '延', '幵': '开', '仟': '千', '來': '来', '挨': '爱', '祢': '你', '糸': '细', '颃': '用', '乳': '用', '借唄': '借呗', '唔': '客', '則': '则', '阔': '可', '叨': '嘛', '花多上': '花多少', '镀': '度', '刭': '到', '冯': '吗', '蔡': '才', '丽': '里', '減': '减', '狂': '款', '錯': '错', '匆': '充', '問': '问', '窃': '切', '贯': '关', '勒': '了', '颌': '额', '敗': '败', '咬': '要', '鈤': '日', '莪': '我', '腨': '用', '吵': '超', '篮': '蓝', '培': '赔', '粑': '把', '躲': '多', '嗎': '吗', '戶': '户', '毎': '每', '呮': '呗', '姑': '过', '胆': '但', '脱': '拖', '胃': '为', '剧': '刷', '吸': '息', '布': '不', '夂': '久', '栋': '冻', '淸': '清', '萌': '能', '愉': '逾', '請': '请', '卅': '啥', '堤': '提', '吱': '支', '禾': '何', '菅': '营', '渝': '逾', '侯': '候', '權': '权', '鼻': '比', '杜': '度', '嗨': '还', '踩': '才', '矿': '款', '珐': '法', 'ma': '吗', '花百': '花呗', '绊': '绑', '甬': '通', '車': '车', '叧': '另', '述': '诉', '査': '查', '瞪': '登', '機': '机', '啲': '的', '設': '设', '綁': '绑', '遲': '迟', '赞': '暂', '粤': '月', '驗': '验', '説': '说', '花花': '花', '坝': '呗', '发呗': '花呗', '虫': '了', '臨': '临', '笫': '第', '廷': '延', '琪': '期', '扥': '等', '谝': '骗', '倩': '欠', '挑': '调', '⑩': '', '/': '', '‘': '', '哩': '', '～±': '', '／': '', '｛': '', ';': '', '：': '', '％': '', '＝': '', '按': '', '喏': '', '>': '', '俺们': '', '》': '', 'Ⅲ': '', '蜂': '', '*': '', '日': '', '=': '', '⑦': '', '～': '', '）': '', ']': '', '。': '', ',': '', '“': '', '}': '', '逼': '', '＞': '', 'ＬＩ': '', '&': '', '(': '', "'": '', '哎哟': '', '数': '', 'sub': '', '！': '', '~': '', '@': '', '∈': '', '咦': '', '?': '', '喔唷': '', '⑤': '', '①': '', 'μ': '', '、': '', 'γ': '', '嗳': '', '』': '', '一': '', '②': '', 'Ｉ': '', 'В': '', '＃': '', '兮': '', '我': '', '［': '', '＋': '', '把': '', 'Ｌ': '', '<': '', '俺': '', '──': '', '⑧': '', '[': '', '④': '', 'sup': '', '哎呀': '', '；': '', '哎': '', 'Ｂ': '', '÷': '', '呗': '', '阿': '', '喔': '', '蚂': '', 'ａ': '', '#': '', 'Ｆ': '', '〔': '', ':': '', '吧': '', '丫': '', '嗯': '', '的': '', '■': '', '”': '', 'Ｔ': '', '＇': '', '《': '', '啐': '', '也': '', '嗬': '', '㈧': '', 'φ': '', '"': '', '↑': '', 'Δ': '', 'Ψ': '', '℃': '', '⑥': '', '〕': '', '+': '', 'exp': '', '＆': '', '罢了': '', '·': '', '″': '', '—': '', '１': '', '〉': '', '...': '', '＊': '', 'Lex': '', '＿': '', '蚁': '', '啊': '', '『': '', '．': '', '【': '', '呃': '', '［－': '', '▲': '', '，': '', '’': '', '｝': '', '（': '', '－': '', '】': '', '×': '', '咧': '', '.': '', '了': '', ')': '', 'Ａ': '', 'Ｘ': '', '咚': '', '］': '', '？': '', '→': '', '③': '', '＜': '', '吧哒': '', '按照': '', '唉': '', '＞λ': '', '———': '', '-': '', 'Ｚ': '', '⑨': '', '～＋': '', '呐': ''}
    for key, value in replace_dict.items():
        sentence = sentence.replace(key, value)
    return sentence

all_data['question1'] = all_data.iloc[:, 1].apply(lambda x: replace_words(x))
all_data['question2'] = all_data.iloc[:, 2].apply(lambda x: replace_words(x))
all_data['jieba_q1'] = all_data.iloc[:, 1].apply(lambda x: jieba_cut(x))
all_data['jieba_q2'] = all_data.iloc[:, 2].apply(lambda x: jieba_cut(x))
all_data['jieba_word_cut_q1'] = all_data.jieba_q1.apply(lambda x: jieba_word_cut(x))
all_data['jieba_word_cut_q2'] = all_data.jieba_q2.apply(lambda x: jieba_word_cut(x))
all_data['first_char_q1'] = all_data.jieba_q1.apply(lambda x: first_char_cut(x))
all_data['first_char_q2'] = all_data.jieba_q2.apply(lambda x: first_char_cut(x))
all_data['word_cut_q1'] = all_data.iloc[:, 1].apply(lambda x: word_cut(x))
all_data['word_cut_q2'] = all_data.iloc[:, 2].apply(lambda x: word_cut(x))


# 特征: 计算（s1的字同时在s2中也出现和s2的字同时在s1中也出现）的比例
def shared_word_proportion(row):
        q1words = {}
        q2words = {}
        for word in row['jieba_word_cut_q1'].split():
            q1words[word] = q1words.get(word, 0) + 1
        for word in row['jieba_word_cut_q2'].split():
            q2words[word] = q2words.get(word, 0) + 1
        n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
        n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
        n_tol = sum(q1words.values()) + sum(q2words.values())
        if 1e-6 > n_tol:
            return 0.
        else:
            return 1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol
        
all_data['shared_word'] = all_data.apply(shared_word_proportion, axis=1, raw=True)        

# 计算所有词的tfidf值
def init_idf(data):
    idf = {}
    q_set = set()
    for index, row in data.iterrows():
        q1 = row['jieba_word_cut_q1']
        q2 = row['jieba_word_cut_q2']
        if q1 not in q_set:
            q_set.add(q1)
            words = q1.split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
        if q2 not in q_set:
            q_set.add(q2)
            words = q2.split()
            for word in words:
                idf[word] = idf.get(word, 0) + 1
    num_docs = len(data)
    for word in idf:
        idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
    return idf

idf = init_idf(all_data)

# 特征：共现词的tfidf比例（跟上面shared_word_proportion类似）
def tfidf_shared_word(row):
    q1words = {}
    q2words = {}
    for word in row['jieba_word_cut_q1'].split():
        q1words[word] = q1words.get(word, 0) + 1
    for word in row['jieba_word_cut_q2'].split():
        q2words[word] = q2words.get(word, 0) + 1
    sum_shared_word_in_q1 = sum([q1words[w] * idf.get(w, 0) for w in q1words if w in q2words])
    sum_shared_word_in_q2 = sum([q2words[w] * idf.get(w, 0) for w in q2words if w in q1words])
    sum_tol = sum(q1words[w] * idf.get(w, 0) for w in q1words) + sum(q2words[w] * idf.get(w, 0) for w in q2words)
    if 1e-6 > sum_tol:
        return 0.
    else:
        return 1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol

all_data['tfidf_shared'] = all_data.apply(tfidf_shared_word, axis=1, raw=True)

# 特征：两个句子的tfidf总和的差值
def tfidf_dif(row):
    q1words = {}
    q2words = {}
    for word in row['jieba_word_cut_q1'].split():
        q1words[word] = q1words.get(word, 0) + 1
    for word in row['jieba_word_cut_q2'].split():
        q2words[word] = q2words.get(word, 0) + 1
    tfidf_q1 = sum([q1words[w] * idf.get(w, 0) for w in q1words])
    tfidf_q2 = sum([q2words[w] * idf.get(w, 0) for w in q2words])
    return abs(tfidf_q1 - tfidf_q2)

all_data['tfidf_dif'] = all_data.apply(tfidf_dif, axis=1, raw=True)

# 特征：长度特征
all_data['word_len1'] = all_data.first_char_q1.apply(lambda x: len(x.split()))
all_data['word_len2'] = all_data.first_char_q2.apply(lambda x: len(x.split()))
all_data['char_len1'] = all_data.word_cut_q1.apply(lambda x: len(x.split()))
all_data['char_len2'] = all_data.word_cut_q2.apply(lambda x: len(x.split()))

# 特征：两个句子长度差
def length_dif(row):
    len_s1 = len(row['word_cut_q1'].split())
    len_s2 = len(row['word_cut_q2'].split())
    len_dif = abs(len_s1 - len_s2)
    return len_dif

all_data['length_dif'] = all_data.apply(length_dif, axis=1, raw=True)

# 特征：两个句子的长度差比例
def length_dif_rate(row):
    len_q1 = len(row['word_cut_q1'].split())
    len_q2 = len(row['word_cut_q2'].split())
    if max(len_q1, len_q2) < 1e-6:
        return 0.0
    else:
        return 1.0 * min(len_q1, len_q2) / max(len_q1, len_q2)

all_data['length_dif_rate'] = all_data.apply(length_dif_rate, axis=1, raw=True)

# 特征：共现词个数
def common_chars(row):
    s1 = set(row['word_cut_q1'].split())
    s2 = set(row['word_cut_q2'].split())
    intersection = s1.intersection(s2)
    return len(intersection)

all_data['common_words'] = all_data.apply(common_chars, axis=1, raw=True)

# 特征：莱文斯顿距离
def levenshtein_dist(row):
    s1 = row['word_cut_q1'].split()
    s2 = row['word_cut_q2'].split()
    if len(s1) < len(s2):
        temp = s1
        s1 = s2
        s2 = temp

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return 1.0 * previous_row[-1]

all_data['levenshtein'] = all_data.apply(levenshtein_dist, axis=1, raw=True)

# 特征：否定词出现情况，一个有否定词，另一个没有，则是不相似的
def neg_word(row):
    neg_words = ['不','非','无','未','不曾','没','没有','别','勿','请勿','不用','无须','并非','毫无','决不','休想','永不','不要','未尝','未曾','毋','莫','从不','从未','从未有过','尚未','一无','并未','尚无','从来不','从没','绝非','远非','切莫','永不','休想','绝不','毫不','不必','禁止','忌','拒绝','杜绝','否','弗','木有']
    s1 = row['jieba_word_cut_q1'].split()
    s2 = row['jieba_word_cut_q2'].split()
    s1_inter = set(s1).intersection(neg_words)
    s2_inter = set(s2).intersection(neg_words)
    if len(s1_inter)>0 and len(s2_inter)>0:
        return 1
    elif len(s1_inter)==0 and len(s2_inter)==0:
        return 1
    else:
        return 0

all_data['neg_word'] = all_data.apply(neg_word, axis=1, raw=True)

# 特征：数字出现情况，一个有数字，另一个没数字，则不相似
def digit_in_sent(row):
    p = re.compile(r'\d+')
    digit_s1 = p.findall(row['question1'])
    digit_s2 = p.findall(row['question2'])
    s1_count = len(digit_s1)
    s2_count = len(digit_s2)
    pair_and = int((0 < s1_count) and (0 < s2_count))
    pair_or = int((0 < s1_count) or (0 < s2_count))
    return [s1_count, s2_count, pair_and, pair_or]

all_data['digit_in_sent'] = all_data.apply(digit_in_sent, axis=1, raw=True)

# 特征：只取出词语的第一个字，来计算相似度
def first_char_shared(row):
    s1 = set(row['first_char_q1'].split())
    s2 = set(row['first_char_q2'].split())
    shared_num = len(s1.intersection(s2))
    return shared_num
def first_char_jaccard(row):
    s1 = set(row['first_char_q1'].split())
    s2 = set(row['first_char_q2'].split())
    intersection = s1.intersection(s2)
    union = s1.union(s2)
    return len(intersection)/len(union)

all_data['1st_shared'] = all_data.apply(first_char_shared, axis=1, raw=True)
all_data['1st_jaccard'] = all_data.apply(first_char_jaccard, axis=1, raw=True)

def generate_powerful_word(data):
    """
    计算数据中词语的影响力，格式如下：
    词语-->[0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
    """
    words_power = {}
    for index, row in data.iterrows():
        label = int(row['is_duplicate'])
        q1_words = row['jieba_word_cut_q1'].split()
        q2_words = row['jieba_word_cut_q2'].split()
        all_words = set(q1_words + q2_words)
        q1_words = set(q1_words)
        q2_words = set(q2_words)
        for word in all_words:
            if word not in words_power:
                words_power[word] = [0. for i in range(7)]
            # 计算出现语句对数量
            words_power[word][0] += 1.
            words_power[word][1] += 1.
            if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                # 计算单侧语句数量
                words_power[word][3] += 1.
                if 0 == label:
                    # 计算正确语句对数量
                    words_power[word][2] += 1.
                    # 计算单侧语句正确比例
                    words_power[word][4] += 1.
            if (word in q1_words) and (word in q2_words):
                # 计算双侧语句数量
                words_power[word][5] += 1.
                if 1 == label:
                    # 计算正确语句对数量
                    words_power[word][2] += 1.
                    # 计算双侧语句正确比例
                    words_power[word][6] += 1.
    for word in words_power:
        # 计算出现语句对比例
        words_power[word][1] /= len(data)
        # 计算正确语句对比例
        words_power[word][2] /= words_power[word][0]
        # 计算单侧语句对正确比例
        if words_power[word][3] > 1e-6:
            words_power[word][4] /= words_power[word][3]
        # 计算单侧语句对比例
        words_power[word][3] /= words_power[word][0]
        # 计算双侧语句对正确比例
        if words_power[word][5] > 1e-6:
            words_power[word][6] /= words_power[word][5]
        # 计算双侧语句对比例
        words_power[word][5] /= words_power[word][0]
    sorted_words_power = sorted(words_power.items(), key=lambda d: d[1][0], reverse=True)
    return sorted_words_power

power_words = generate_powerful_word(all_data)

def init_powerful_word_dside(pword, thresh_num, thresh_rate):
    pword_dside = []
    pword = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)
    pword_sort = sorted(pword, key=lambda d: d[1][6], reverse=True)
    pword_dside.extend(map(lambda x: x[0], filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))
    return pword_dside
power_word_dside = init_powerful_word_dside(power_words, 20000, 0.4)

def power_dside(row):
    tags = []
    q1_words = row['jieba_word_cut_q1'].split()
    q2_words = row['jieba_word_cut_q2'].split()
    for word in power_word_dside:
        if (word in q1_words) and (word in q2_words):
            tags.append(1.0)
        else:
            tags.append(0.0)
    return tags

all_data['power_dside'] = all_data.apply(power_dside, axis=1, raw=True)

def init_powerful_word_oside(pword, thresh_num, thresh_rate):
    pword_oside = []
    pword = filter(lambda x: x[1][0] * x[1][3] >= thresh_num, pword)
    pword_oside.extend(map(lambda x: x[0], filter(lambda x: x[1][4] >= thresh_rate, pword)))
    return pword_oside
power_word_oside = init_powerful_word_oside(power_words, 30000, 0.5)

def power_oside(row):
    tags = []
    q1_words = set(row['jieba_word_cut_q1'].split())
    q2_words = set(row['jieba_word_cut_q2'].split())
    for word in power_word_oside:
        if (word in q1_words) and (word not in q2_words):
            tags.append(1.0)
        elif (word not in q1_words) and (word in q2_words):
            tags.append(1.0)
        else:
            tags.append(0.0)
    return tags

all_data['power_oside'] = all_data.apply(power_oside, axis=1, raw=True)

# 特征：句子在语料中的重复次数
def duplicate_num():
    dup_num = {}
    for index, row in all_data.iterrows():
        q1 = row['question1']
        q2 = row['question2']
        dup_num[q1] = dup_num.get(q1, 0) + 1
        if q1 != q2:
            dup_num[q2] = dup_num.get(q2, 0) + 1
    return dup_num
dup_num = duplicate_num()

def duplicate_sent(row):
    s1 = row['question1']
    s2 = row['question2']
    s1_num = dup_num[s1]
    s2_num = dup_num[s2]
    return [s1_num, s2_num, max(s1_num, s2_num), min(s1_num, s2_num)]

all_data['duplicate_sent'] = all_data.apply(duplicate_sent, axis=1, raw=True)

# 距离计算函数！
def jaccard_coef(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return float(len(A.intersection(B)) / len(A.union(B)))

def dice_dist(A, B):
    if not isinstance(A, set):
        A = set(A)
    if not isinstance(B, set):
        B = set(B)
    return (2.0 * float(len(A.intersection(B)))) / (len(A) + len(B))


# 计算 N-gram.
def unigrams(words):
    assert type(words) == list
    return words

def bigrams(words, join_string, skip=0):
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L - 1):
            for k in range(1, skip + 2):
                if i + k < L:
                    lst.append(join_string.join([words[i], words[i + k]]))
    else:
        # set it as unigram
        lst = unigrams(words)
    return lst

def trigrams(words, join_string, skip=0):
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in range(L - 2):
            for k1 in range(1, skip + 2):
                for k2 in range(1, skip + 2):
                    if i + k1 < L and i + k1 + k2 < L:
                        lst.append(join_string.join([words[i], words[i + k1], words[i + k1 + k2]]))
    else:
        # set it as bigram
        lst = bigrams(words, join_string, skip)
    return lst

def fourgrams(words, join_string):
    assert type(words) == list
    L = len(words)
    if L > 3:
        lst = []
        for i in xrange(L - 3):
            lst.append(join_string.join([words[i], words[i + 1], words[i + 2], words[i + 3]]))
    else:
        # set it as trigram
        lst = trigrams(words, join_string)
    return lst

def ngrams(words, ngram, join_string=" "):
    if ngram == 1:
        return unigrams(words)
    elif ngram == 2:
        return bigrams(words, join_string)
    elif ngram == 3:
        return trigrams(words, join_string)
    elif ngram == 4:
        return fourgrams(words, join_string)
    elif ngram == 12:
        unigram = unigrams(words)
        bigram = [x for x in bigrams(words, join_string) if len(x.split(join_string)) == 2]
        return unigram + bigram
    elif ngram == 123:
        unigram = unigrams(words)
        bigram = [x for x in bigrams(words, join_string) if len(x.split(join_string)) == 2]
        trigram = [x for x in trigrams(words, join_string) if len(x.split(join_string)) == 3]
        return unigram + bigram + trigram

# 特征：n-gram jaccard系数
def ngram_jaccard(row):
    q1_words = row['jieba_word_cut_q1'].split()
    q2_words = row['jieba_word_cut_q2'].split()
    fs = list()
    for n in range(1, 4):
        q1_ngrams = ngrams(q1_words, n)
        q2_ngrams = ngrams(q2_words, n)
        fs.append(jaccard_coef(q1_ngrams, q2_ngrams))
    return fs

all_data['ngram_jaccard'] = all_data.apply(ngram_jaccard, axis=1, raw=True)

# 特征：n-gram dice系数
def ngram_dice(row):
    q1_words = row['jieba_word_cut_q1'].split()
    q2_words = row['jieba_word_cut_q2'].split()
    fs = list()
    for n in range(1, 4):
        q1_ngrams = ngrams(q1_words, n)
        q2_ngrams = ngrams(q2_words, n)
        fs.append(dice_dist(q1_ngrams, q2_ngrams))
    return fs

all_data['ngram_dice'] = all_data.apply(ngram_dice, axis=1, raw=True)

lgb_label = all_data['is_duplicate']
lgb_data = all_data.drop(columns=['id', 'question1', 'question2', 'is_duplicate', 'jieba_q1', 'jieba_q2',
                                  'jieba_word_cut_q1', 'jieba_word_cut_q2', 'first_char_q1', 'first_char_q2',
                                  'word_cut_q1', 'word_cut_q2'])

lgb_data['digit_in_sent'] = lgb_data.digit_in_sent.apply(lambda x: ','.join(map(str, x)))
digit_sent = lgb_data['digit_in_sent'].str.split(',', expand=True).rename(columns = lambda x: 'digit_sent_' + str(x+1))
digit_sent = digit_sent.apply(pd.to_numeric)
lgb_data = pd.concat([lgb_data, digit_sent], axis=1)
lgb_data.drop(['digit_in_sent'], axis=1, inplace=True)

lgb_data['power_dside'] = lgb_data.power_dside.apply(lambda x: ','.join(map(str, x)))
power_d = lgb_data['power_dside'].str.split(',', expand=True).rename(columns = lambda x: 'power_d_' + str(x+1))
power_d = power_d.apply(pd.to_numeric)
lgb_data = pd.concat([lgb_data, power_d], axis=1)
lgb_data.drop(['power_dside'], axis=1, inplace=True)

lgb_data['power_oside'] = lgb_data.power_oside.apply(lambda x: ','.join(map(str, x)))
power_o = lgb_data['power_oside'].str.split(',', expand=True).rename(columns = lambda x: 'power_o_' + str(x+1))
power_o = power_o.apply(pd.to_numeric)
lgb_data = pd.concat([lgb_data, power_o], axis=1)
lgb_data.drop(['power_oside'], axis=1, inplace=True)

lgb_data['duplicate_sent'] = lgb_data.duplicate_sent.apply(lambda x: ','.join(map(str, x)))
dup_sent = lgb_data['duplicate_sent'].str.split(',', expand=True).rename(columns = lambda x: 'dup_sent_' + str(x+1))
dup_sent = dup_sent.apply(pd.to_numeric)
lgb_data = pd.concat([lgb_data, dup_sent], axis=1)
lgb_data.drop(['duplicate_sent'], axis=1, inplace=True)

lgb_data['ngram_jaccard'] = lgb_data.ngram_jaccard.apply(lambda x: ','.join(map(str, x)))
ngram_jac = lgb_data['ngram_jaccard'].str.split(',', expand=True).rename(columns = lambda x: 'ngram_jac_' + str(x+1))
ngram_jac = ngram_jac.apply(pd.to_numeric)
lgb_data = pd.concat([lgb_data, ngram_jac], axis=1)
lgb_data.drop(['ngram_jaccard'], axis=1, inplace=True)

lgb_data['ngram_dice'] = lgb_data.ngram_dice.apply(lambda x: ','.join(map(str, x)))
ngram_di = lgb_data['ngram_dice'].str.split(',', expand=True).rename(columns = lambda x: 'ngram_di_' + str(x+1))
ngram_di = ngram_di.apply(pd.to_numeric)
lgb_data = pd.concat([lgb_data, ngram_di], axis=1)
lgb_data.drop(['ngram_dice'], axis=1, inplace=True)

lgb_data.to_csv('./data/feature_table.csv', index=False)
print('总特征个数为：', len(lgb_data.columns.values.tolist()))


'''LightGBM'''
x = lgb_data.values
y = lgb_label.values
x, x_test, y, y_test = train_test_split(x, y, test_size=0.15, random_state=123, stratify=y, shuffle=True)

# F1值度量方法
def threshold(i):
    if i > 0.40:
        return 1.0
    else:
        return 0.0
def f1_metric(y_pred, train_data):
    y_true = train_data.get_label()
    #y_pred = np.round(y_pred)
    y_pred = list(map(threshold, y_pred))
    return 'f1_score', f1_score(y_true, y_pred), True

categorical_features = lgb_data.columns.values.tolist()
lgb_train = lgb.Dataset(x, label=y, feature_name=categorical_features, categorical_feature=categorical_features, free_raw_data=False)
lgb_test = lgb.Dataset(x_test, label=y_test)

parameters = {'application': 'binary',
              'objective': 'binary',
              'is_unbalance': 'true',
              'boosting': 'gbdt',
              'num_leaves': 31,
              'feature_fraction': 0.5,
              'bagging_fraction': 0.5,
              'bagging_freq': 20,
              'learning_rate': 0.05,
              'verbose': 0
             }
parameters['metric'] = ['binary_logloss']
# parameters['metric'] = ['None']

model = lgb.train(params=parameters,
                  train_set=lgb_train,
                  valid_sets=lgb_test,
                  num_boost_round=5000,
                  early_stopping_rounds=100,
                  feval=f1_metric)