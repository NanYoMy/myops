
'''
4,8,16,32,32
32,16,8,4,4

'''
def generate_filter_pair(filter):
    forward_filter=filter

    filter.reverse()
    filter = filter[1:]
    filter.append(filter[-1])
    return forward_filter,filter