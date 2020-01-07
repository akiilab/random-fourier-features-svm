import os
import random
import pickle
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('rows', type=int, help='the number of row')
    parser.add_argument('cols', type=int, help='the number of column')
    parser.add_argument('-o', '--output', help='(option) output file. default `a`')
    parser.add_argument('-d', '--dir', help='(option) output file directory. default `case`')
    args = parser.parse_args()

    if args.output == None:
        args.output = 'a'
    if args.dir == None:
        args.dir = 'cases'

    return args

def generate_random_array(n):
    s, arr = 0.0, []
    for _ in range(n):
        arr.append(s)
        s = float(format(s + random.uniform(0, 1) * 0.9 + 0.1, '.2f'))
    return arr

def main():
    args = arg_parse()

    if not os.path.exists(args.dir + "/data"):
        os.makedirs(args.dir + "/data")
    if not os.path.exists(args.dir + "/sample"):
        os.makedirs(args.dir + "/sample")
    if not os.path.exists(args.dir + "/target"):
        os.makedirs(args.dir + "/target")

    fd = open(args.dir + "/data/" + args.output + ".csv","w+")
    ft = open(args.dir + "/target/" + args.output + ".csv","w+")
    fs = open(args.dir + "/sample/" + args.output + ".pkl","wb")

    fd.write('"(%d %d)","overflow_H","overflow_V","rmgTrack_H","rmgTrack_V","totTrack_H","totTrack_V"\n' % (args.rows, args.cols))
    ft.write('"(%d %d)","Short_VioNum"\n' % (args.rows, args.cols))

    rows = generate_random_array(args.rows)
    cols = generate_random_array(args.cols)

    samples = []

    for i, x in enumerate(rows):
        for j, y in enumerate(cols):
            fd.write("\"{:.2f} {:.2f}\",{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}\n".format(
                *[x, y] + [random.uniform(0, 10) for _ in range(6)]))

            ft.write("\"%.2f %.2f\",%.1f\n" %
                    (x, y, random.getrandbits(1)))

            if random.getrandbits(1):
                samples.append((j, i))

    fd.close
    ft.close
    pickle.dump(samples, file=fs)
    print("%dx%d data are generated and save at %s/{data,target,sample}/%s " %
            (args.rows, args.cols, args.dir, args.output))

if __name__ == '__main__':
    main()

