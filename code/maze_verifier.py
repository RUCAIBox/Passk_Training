import re

def extract_answer_maze(s):
    if ('<answer>' in s and "</answer>" in s):
        s = s.split("<answer>")[-1].strip().split("</answer>")[0].strip()
    ans = s.split("boxed")
    if len(ans) == 1:
        return s
    ans = ans[-1]
    if len(ans) == 0:
        return ""
    try:
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
    except:
        return ""
    return a


def compute_score(solution, maze):
    solution = extract_answer_maze(solution)
    solution = solution.upper()
    maze = maze.strip().split('\n')
    n = len(maze)
    m = len(maze[0])

    def find_st(maze):
        for i in range(n):
            for j in range(m):
                if (maze[i][j] == 'S'):
                    return i, j
        assert(False)

    x, y = find_st(maze)
    for step in solution:
        if (step == 'L'):
            y = y - 1
        elif (step == 'R'):
            y = y + 1
        elif (step == 'U'):
            x = x - 1
        elif (step == 'D'):
            x = x + 1
        else:
            continue
        
        if (x < 0 or x >= n):
            return 0
        if (y < 0 or y >= m):
            return 0
        if (maze[x][y] == '*'):
            return 0
    
    if (maze[x][y] == 'E'):
        return 1
    else:
        return 0