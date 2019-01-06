class Solution:

    def make(self, mark):
        r = [['.' for _ in range(len(mark))] for _ in range(len(mark))]
        for i in mark:
            r[i][mark[i]] = 'Q'
        for k, v in enumerate(r):
            r[k] = ''.join(v)
        return r

    def recursive(self, mark, cur, ret):
        """

        :param mark: 放置的列
        :param cur: 所在行
        :param ret:
        :return:
        """

        if cur == len(mark):
            ret.append(self.make(mark))
            return

        for i in range(len(mark)):
            mark[cur], down = i, True
            for j in range(cur):
                if mark[j] == i or abs(i-mark[j]) == cur - j:
                    down = False
                    break
            if down:
                self.recursive(mark, cur+1, ret)

    def solveNQueens(self, n):

        ret = []
        self.recursive([None]*n, 0, ret)
        return ret
if __name__ == "__main__":
    print(Solution().solveNQueens(10))