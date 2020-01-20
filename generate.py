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

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    args = arg_parse()

    mkdir(args.dir + "/data")
    mkdir(args.dir + "/target")
    mkdir(args.dir + "/train_sample")
    mkdir(args.dir + "/valid_sample")

    fd_data = open(args.dir + "/data/" + args.output + ".csv","w+")
    fd_target = open(args.dir + "/target/" + args.output + ".csv","w+")
    fd_train = open(args.dir + "/train_sample/" + args.output + ".pkl","wb")
    fd_valid = open(args.dir + "/valid_sample/" + args.output + ".pkl","wb")

    fd_data.write('"(%d %d)","overflow_H","overflow_V","rmgTrack_H","rmgTrack_V","totTrack_H","totTrack_V"\n' % (args.rows, args.cols))
    fd_target.write('"(%d %d)","Short_VioNum"\n' % (args.rows, args.cols))

    rows = generate_random_array(args.rows)
    cols = generate_random_array(args.cols)

    train_samples = []
    valid_samples = []

    for i, x in enumerate(rows):
        for j, y in enumerate(cols):
            fd_data.write("\"{:.2f} {:.2f}\",{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f}\n".format(
                *[x, y] + [random.uniform(0, 10) for _ in range(6)]))

            fd_target.write("\"%.2f %.2f\",%.1f\n" %
                    (x, y, random.getrandbits(1)))

            rand = random.uniform(0, 1)
            if rand > 0.8:
                valid_samples.append((i, j))
            else:
                train_samples.append((i, j))

    fd_data.close
    fd_target.close
    pickle.dump(train_samples, file=fd_train)
    pickle.dump(valid_samples, file=fd_valid)
    print("%dx%d data are generated and save at %s/{data,target,train_sample,valid_sample}/%s " %
            (args.rows, args.cols, args.dir, args.output))

if __name__ == '__main__':
    main()

