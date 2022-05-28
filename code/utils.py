'''
Author: your name
Date: 2022-03-21 10:54:29
LastEditTime: 2022-04-01 13:41:21
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /huizhong/FM/utils.py
'''
def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat_name': feat}

def getFeature(sample_num):
    if sample_num == 2000:
        feat = [{'feat_name': 'C1', 'feat_num': 1454, 'embed_dim': 8},
                {'feat_name': 'C2', 'feat_num': 576, 'embed_dim': 8},
                {'feat_name': 'C3', 'feat_num': 6131817, 'embed_dim': 8},
                {'feat_name': 'C4', 'feat_num': 1450332, 'embed_dim': 8},
                {'feat_name': 'C5', 'feat_num': 305, 'embed_dim': 8},
                {'feat_name': 'C6', 'feat_num': 24, 'embed_dim': 8},
                {'feat_name': 'C7', 'feat_num': 12449, 'embed_dim': 8},
                {'feat_name': 'C8', 'feat_num': 632, 'embed_dim': 8},
                {'feat_name': 'C9', 'feat_num': 3, 'embed_dim': 8},
                {'feat_name': 'C10', 'feat_num': 82370, 'embed_dim': 8},
                {'feat_name': 'C11', 'feat_num': 5598, 'embed_dim': 8},
                {'feat_name': 'C12', 'feat_num': 5135114, 'embed_dim': 8},
                {'feat_name': 'C13', 'feat_num': 3192, 'embed_dim': 8},
                {'feat_name': 'C14', 'feat_num': 26, 'embed_dim': 8},
                {'feat_name': 'C15', 'feat_num': 14304, 'embed_dim': 8},
                {'feat_name': 'C16', 'feat_num': 3457190, 'embed_dim': 8},
                {'feat_name': 'C17', 'feat_num': 10, 'embed_dim': 8},
                {'feat_name': 'C18', 'feat_num': 5454, 'embed_dim': 8},
                {'feat_name': 'C19', 'feat_num': 2158, 'embed_dim': 8},
                {'feat_name': 'C20', 'feat_num': 4, 'embed_dim': 8},
                {'feat_name': 'C21', 'feat_num': 4398024, 'embed_dim': 8},
                {'feat_name': 'C22', 'feat_num': 18, 'embed_dim': 8},
                {'feat_name': 'C23', 'feat_num': 15, 'embed_dim': 8},
                {'feat_name': 'C24', 'feat_num': 213138, 'embed_dim': 8},
                {'feat_name': 'C25', 'feat_num': 105, 'embed_dim': 8},
                {'feat_name': 'C26', 'feat_num': 118826, 'embed_dim': 8},
                {'feat_name': 'I1', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I2', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I3', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I4', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I5', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I6', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I7', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I8', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I9', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I10', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I11', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I12', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I13', 'feat_num': 100, 'embed_dim': 8}]
    if sample_num == 1000:
    # train 1000w
        feat = [{'feat_name': 'C1', 'feat_num': 1396, 'embed_dim': 8},
                {'feat_name': 'C2', 'feat_num': 553, 'embed_dim': 8},
                {'feat_name': 'C3', 'feat_num': 2594031, 'embed_dim': 8},
                {'feat_name': 'C4', 'feat_num': 698469, 'embed_dim': 8},
                {'feat_name': 'C5', 'feat_num': 290, 'embed_dim': 8},
                {'feat_name': 'C6', 'feat_num': 23, 'embed_dim': 8},
                {'feat_name': 'C7', 'feat_num': 12048, 'embed_dim': 8},
                {'feat_name': 'C8', 'feat_num': 608, 'embed_dim': 8},
                {'feat_name': 'C9', 'feat_num': 3, 'embed_dim': 8},
                {'feat_name': 'C10', 'feat_num': 65156, 'embed_dim': 8},
                {'feat_name': 'C11', 'feat_num': 5309, 'embed_dim': 8},
                {'feat_name': 'C12', 'feat_num': 2186509, 'embed_dim': 8},
                {'feat_name': 'C13', 'feat_num': 3128, 'embed_dim': 8},
                {'feat_name': 'C14', 'feat_num': 26, 'embed_dim': 8},
                {'feat_name': 'C15', 'feat_num': 12750, 'embed_dim': 8},
                {'feat_name': 'C16', 'feat_num': 1537323, 'embed_dim': 8},
                {'feat_name': 'C17', 'feat_num': 10, 'embed_dim': 8},
                {'feat_name': 'C18', 'feat_num': 5002, 'embed_dim': 8},
                {'feat_name': 'C19', 'feat_num': 2118, 'embed_dim': 8},
                {'feat_name': 'C20', 'feat_num': 4, 'embed_dim': 8},
                {'feat_name': 'C21', 'feat_num': 1902327, 'embed_dim': 8},
                {'feat_name': 'C22', 'feat_num': 17, 'embed_dim': 8},
                {'feat_name': 'C23', 'feat_num': 15, 'embed_dim': 8},
                {'feat_name': 'C24', 'feat_num': 135789, 'embed_dim': 8},
                {'feat_name': 'C25', 'feat_num': 94, 'embed_dim': 8},
                {'feat_name': 'C26', 'feat_num': 84305, 'embed_dim': 8},
                {'feat_name': 'I1', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I2', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I3', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I4', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I5', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I6', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I7', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I8', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I9', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I10', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I11', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I12', 'feat_num': 100, 'embed_dim': 8},
                {'feat_name': 'I13', 'feat_num': 100, 'embed_dim': 8}]
    return feat