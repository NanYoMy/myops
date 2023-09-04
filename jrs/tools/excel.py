import xlrd
import xlwt
import os
from tools.dir import  mkdir_if_not_exist

def read_excel(path,reduce_mean_std=False):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_index(0)
    rownum=sheet.nrows
    colnum=sheet.ncols
    dict={}
    for i in range(colnum):
        list=[]
        for j in range(1,rownum):
           list.append(sheet.cell_value(j,i))
        if reduce_mean_std==True:
            dict[sheet.cell_value(0,i)]=list[:-2]
        else:
            dict[sheet.cell_value(0, i)] = list
    return dict

def read_excelV2(path,reduce_mean_std=False):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_index(0)
    rownum=sheet.nrows
    colnum=sheet.ncols
    dict={}
    for i in range(colnum):
        list=[]
        for j in range(1,rownum):
           list.append(sheet.cell_value(j,i))
        if reduce_mean_std==True:
            dict[sheet.cell_value(0,i)]=list[:-2]
        else:
            tmp=list[:-2]
            tmp.append(np.mean(tmp))
            dict[sheet.cell_value(0, i)] = tmp
    return dict


from decimal import Decimal,ROUND_HALF_EVEN,Context
def r4(input_value,tmp=4):
    try:
        ctx = Context(prec=tmp, rounding=ROUND_HALF_EVEN)
        a=str(ctx.create_decimal(input_value))
        return float(a)
    except:
        return -1

def write_excel(path,map):
    writebook = xlwt.Workbook(encoding = 'utf-8')
    sheet = writebook.add_sheet("all")

    for i,key in enumerate(map.keys()):
        list=map[key]
        sheet.write(0,i,key)
        for j,item in enumerate(list):
            item=r4(item)
            sheet.write(j+1,i,item)

    writebook.save(path)
import numpy as np
def write_array(path, id, array,need_static=True):

    if os.path.exists(path):
        map=read_excel(path)
    else:
        mkdir_if_not_exist(os.path.dirname(path))
        map={}

    #add static infomration such mean and std
    if need_static==True:
        np_array=np.array(array)

        if id.find("hd")>0 or id.find("asd")>0:
            mean=np.mean(np_array)
            std=np.std(np_array)
        else:
            mean=np.mean(np_array)*100
            std=np.std(np_array)*100
        array.append(mean)
        array.append(std)


    map[id]=array
    write_excel(path,map)

def write_dict(path,new_map):
    if os.path.exists(path):
        map=read_excel(path)
    else:
        mkdir_if_not_exist(os.path.dirname(path))
        map={}
    new_map={**new_map,**map}
    write_excel(path,new_map)

def outpu2excel(path, id, array):

    if os.path.exists(path):
        map=read_excel(path)
    else:
        mkdir_if_not_exist(os.path.dirname(path))
        map={}
    map[id]=array
    write_excel(path,map)


if __name__ == "__main__":
    write_array('../../outputs/result/result.xls', 'mmwhs-ct-mr-fold-4-ds', [1.0, 0.2, 3])
    write_array('../../outputs/result/result.xls', 'mmwhs-ct-mr-fold-2-hd', [1.0, 0.2, 3])
    write_array('../../outputs/result/result.xls', 'mmwhs-ct-mr-fold-3', [1.0, 0.2, 3])
