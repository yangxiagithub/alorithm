// 一. 排序算法

/** 1. 冒泡排序
 * 稳定性：稳定
 * @param {*} arr 
 */
function bubble(arr) {
    const length = arr.length;
    for(let round = 0; round < length - 1; round++) {
        for(let current = 0; current < length - 1 - round; current++) {
            if(arr[current] > arr[current + 1]) {
                let temp = arr[current + 1];
                arr[current + 1] = arr[current];
                arr[current] = temp;
            }
        }
    }
    return arr;
}

/** 2. 选择排序
 * 稳定性：不稳定 如：58529
 * @param {*} arr 
 */
function choose(arr) {
    const length = arr.length;
    for(let round = 0; round < length - 1; round++) {
        for(let current = round + 1; current < length; current++) {
            if(arr[round] > arr[current]) {
                let temp = arr[current];
                arr[current] = arr[round];
                arr[round] = temp;
            }
        }
    }
    return arr;
}

/** 3. 插入排序
 * 稳定性：稳定
 * @param {*} arr 
 */
function insert(arr) {
    const length = arr.length;
    for (let basePosition = 0; basePosition < length; basePosition++) {
        let baseValue = arr[basePosition];
        let position = basePosition;
        while (position > 0 && arr[position - 1] > baseValue) {
            arr[position] = arr[position - 1];
            position--;
        }
        arr[position] = baseValue;
    }
    return arr;
}

/** 4. 归并排序
 * 稳定性： 稳定
 * @param {*} arr 
 */
function mergeSort(arr) {
    const length = arr.length;
    if(length <= 1) return arr;
    const mid = Math.floor(length / 2);
    const left = arr.slice(0, mid);
    const right = arr.slice(mid);
    return merge(mergeSort(left), mergeSort(right));
}

function merge(left, right) {
    let result = [];
    while(left.length && right.length) {
        if(left[0] < right[0]) {
            result.push(left.shift());
        } else {
            result.push(right.shift());
        }
    }
    return result.concat(left, right);
}

/** 5.快排
 * 稳定性：不稳定
 * @param {*} arr 
 */
function quickSort(arr) {
    const length = arr.length;
    if(length <= 1) return arr;
    const mid = Math.floor(length / 2);
    const midValue = arr.splice(mid, 1)[0];
    let left = [];
    let right = [];
    for(let i = 0; i < arr.length; i++) {
        if(arr[i] > midValue) {
            right.push(arr[i]);
        } else {
            left.push(arr[i]);
        }
    }
    return quickSort(left)
        .concat([midValue])
        .concat(quickSort(right));
}
 
/** 6. 堆排序
 * 构建堆的时候从下往上调整
 * 调整堆的时候从上往下调整
 * 不稳定
 * @param {*} elements 
 */
function heapSort(elements) {
    buildHeap(elements);
    for(let i = elements.length - 1; i > 0; i--) {
        let swap = elements[i];
        elements[i] = elements[0];
        elements[0] = swap;
        headAjust(elements, 0, i);
    }
}

function buildHeap(elements) {
    for (let i = Math.ceil(elements.length / 2); i >= 0; i--) {
        headAjust(elements, i, elements.length);
    }
}

function headAjust(elements, pos, len) {
    let leftChildPosition = pos * 2 + 1;
    let rightChildPosition = pos * 2 + 2;
    let maxChildPosition = leftChildPosition;
    if (leftChildPosition >= len) {
        return;
    }
    if (rightChildPosition < len && elements[leftChildPosition] < elements[rightChildPosition]) {
        maxChildPosition = rightChildPosition;
    }
    if(elements[pos] < elements[maxChildPosition]) {
        const temp = elements[pos];
        elements[pos] = elements[maxChildPosition];
        elements[maxChildPosition] = temp;
        headAjust(elements, maxChildPosition, len);
    }
}

// 二. 数组去重

var arr = [1, 1, 'true', 'true', true, true, 15, 15, false, false, undefined, undefined, null, null, NaN, NaN, 'NaN', 0, 0, 'a', 'a', {}, {}];

/**
 * 1. 利用ES6 Set
 * 缺点：无法去重空对象{}
 */
function duplicate1(arr) {
    return Array.from(new Set(arr))
}

function duplicate11() {
    return [...new Set(arr)];
}

/**
 * 2. 利用splice去重
 * 确定：{} 和 NaN 无法去重，因为{} !== {}, NaN !== NaN
 */

function depulicate2(arr) {
    const copy = arr.slice();
    for (var i = 0; i < copy.length; i++) {
        for (var j = i + 1; j < copy.length; j++) {
            if (copy[i] === copy[j]) {
                copy.splice(j, 1);
                j--;
            }
        }
    }
    return copy;
}

/** 
 * 3. 利用将数组内容放进另一个数组里
 */
function depulicate3(arr) {
    const otherArr = [];
    arr.forEach((item) => {
        if (!~otherArr.indexOf(item)) {
            otherArr.push(item)
        }
    });
    return otherArr;
}

// 三. 查找算法

/**
 * 1. 二分查找
 */
function binarySerch(arr, target) {
    var end = arr.length - 1;
    var start = 0;
    while (start < end) {
        var mid = Math.floor((end + start) / 2);
        if (arr[mid] === target) {
            return mid;
        } else if (arr[mid] < target) {
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }
    return -1;
}

// 四. 数组Array 相关习题

/** 
 * 求两个数之和为特定数字
 */
var twoSum = function (nums, target) {
    for (let i = 0; i < nums.length; i++) {
        var otherNum = target - nums[i];
        var index = nums.indexOf(otherNum);
        if (index > -1 && index !== i) {
            return [i, index]
        }
    }
    return []
};
/**
 * 求两个数之和为特定数字
 * 双指针
 */
var twoSum = function (numbers, target) {
    var i = 0;
    var j = numbers.length - 1;
    while (i<= numbers.length - 1 && j >= 0 && i<j) {
        if (numbers[i] + numbers[j] > target) {
            j--;
        } else if (numbers[i] + numbers[j] < target) {
            i++;
        } else {
            return [i+1,j+1];
        }
    }
};

/**  
 * 三个数之和为0
*/
var threeSum = function (nums) {
    var result = []
    if (nums.length < 3) return result;
    nums.sort((a, b) => {
        return a - b;
    });
    for (var i = 0; i < nums.length - 2; i++) {
        if (nums[i] > 0) continue;
        if (i > 0 && nums[i] === nums[i - 1]) continue;
        var j = i + 1, k = nums.length - 1;
        while (j < k) {
            if (nums[i] + nums[j] + nums[k] === 0) {
                if (nums[i] === nums[j] && nums[j] === 0) {
                    return;
                }
                result.push([nums[i], nums[j], nums[k]]);
                j++;
                k--;
                while (j < k && nums[j] === nums[j - 1]) {
                    j++;
                }
                while (j < k && nums[k] === nums[k + 1]) {
                    k--;
                }
            } else if (nums[i] + nums[j] + nums[k] > 0) {
                k--;
            } else {
                j++;
            }
        }
    }
    return result;
};
/**
 * 售卖股票的最佳时机：可以多次售卖
 */
var maxProfit = function (prices) {
    if (!prices || !prices.length ||prices.length === 1) return 0;
    var sum = 0;
    prices.reduce((pre, item) => {
        item > pre ? sum += item - pre : '';
        return item;
    });
    return sum;
};

/** 
 * 最佳售卖股票时机：只能一次售卖
 * （只需要看y轴）
 * [7,1,5,3,6,4] =》 5
 * [7,6,4,3,1] =》 0
 */
var maxProfit = function (prices) {
    if (prices.length <= 1) return 0;
    let diff = 0;
    prices.reduce((acc, next) => {
        if (acc > next) {
            return next;
        } else {
            if (next - acc > diff) {
                diff = next - acc;
            }
            return acc;
        }

    })
    return diff
};


/** 
 * 最大面积（需要看x轴和y轴）
*/
var maxArea = function (height) {
    var max = 0;
    for (var i = 0, j = height.length; i < j) {
        max = Math.max(max, Math.min(height[i], height[j]) * (j - i));
        if (height[i] < height[j]) {
            i++;
        } else if (height[i] > height[j]) {
            j--;
        } else {
            i++;
            j--;
        }
    }
    return max;
};

/** 
 * 除自身之外的数组乘机
*/
var productExceptSelf = function (nums) {
    var prefix = [];
    var postfix = [];
    var base = 1;
    for (var i = 0; i < nums.length; i++) {
        let count = nums[i - 1] !== undefined ? nums[i - 1] : 1;
        prefix.push(base *= count)
    }
    base = 1;
    for (var i = nums.length - 1; i >= 0; i--) {
        let count = nums[i + 1] !== undefined ? nums[i + 1] : 1;
        postfix.unshift((base *= count));
    }
    var result = [];
    for (var i = 0; i < nums.length; i++) {
        result.push(prefix[i] * postfix[i])
    }
    return result;
};

/** 
 * 最大子序列乘机(硬算)
 */
var maxSubProduct = function (arr) {
    if (arr.length < 1) return;
    var max = arr[0];
    for (var i = 0; i < arr.length; i++) {
        var row = [];
        row[i] = arr[i];
        if (max < arr[i]) {
            max = arr[i];
        }
        for (var j = i + 1; j < arr.length; j++) {
            row[j] = row[j - 1] * arr[j];
            if (row[j] > max) {
                max = row[j];
            }
        }
    }
    return max;
}
/**
 * 最大子序列乘机(存下最大值和最小值)
 */
var maxSubProduct = function (arr) {
    if (arr.length < 1) return;
    var max = arr[0], min = arr[0], output = arr[0];
    for (var i = 1; i < arr.length; i++) {
        var preMax = max;

        max = Math.max(arr[i], max * arr[i], min * arr[i])
        min = Math.min(arr[i], preMax * arr[i], min * arr[i]);

        output = Math.max(max, output);
    }
    return output;
}

/** 
 * 最大子序列和
*/
var maxSubArraySum = function (nums) {
    if (nums.length < 1) return 0;
    var largeSum = nums[0];
    var lastSum = nums[0];
    for (var i = 1; i < nums.length; i++) {
        lastSum = lastSum > 0 ? (lastSum + nums[i]) : nums[i];
        if (lastSum > largeSum) {
            largeSum = lastSum;
        }
    }
    return largeSum;
}

/**
 * 有序数组从某一个位置翻转，查找目标数字的位置（和二分查找有点像）
 */
nums = [4, 5, 6, 7, 0, 1, 2], target = 0
var search = function (nums, target) {
    var start = 0, end = nums.length;
    return searchIndex(nums, target, start, end);
};
function searchIndex(nums, target, start, end) {
    // if (start >= end - 1) {
    //     if (target === nums[start]) return start;
    //     return -1;
    // }
    var mid = Math.floor(start + (end - start) / 2);
    if(nums[mid] === target) return mid;
    var left = nums.slice(start, mid);
    var right = nums.slice(mid, end);
    // 左边有序
    if (left[0] <= left[left.length - 1]) {
        if (left[0] <= target && target <= left[left.length - 1]) {
            return searchIndex(nums, target, start, mid);
        }
        return searchIndex(nums, target, mid, end);
    }
    // 右边有序
    else if(right[0] <= right[right.length - 1]) {
        if (right[0] <= target && target <= right[right.length - 1]) {
            return searchIndex(nums, target, mid, end);
        }
        return searchIndex(nums, target, start, mid);
    }
    return -1;
}

/**
 * 移动窗口最大值
 */
var maxWindow = function(arr, k) {
    if(arr.length < k) return 0;
    var result = [];
    for(var i = 0; i + k <= arr.length; i++) {
        result.push(Math.max.apply(null, arr.slice(i, i+k)));
    }
    return result;
}

/** 
 * 用栈实现括号匹配
*/
var isValid = function (s) {
    var couple = {
        '{': '}',
        '[': ']',
        '(': ')'
    }
    var arr = s.split('');
    var stack = [];
    while (arr.length) {
        if (couple[stack[stack.length - 1]] === arr[0]) {
            stack.pop();
            arr.shift();
        } else {
            stack.push(arr.shift());
        }
    }
    return !stack.length;
};
/**
 * 数组中元素与下一个比它大的元素之间的距离
 */
var dailyTemperatures = function (arr) {
    var result = [];
    for (let i = 0; i < arr.length; i++) {
        var getHigher = false;
        for (let j = i + 1; j < arr.length; j++) {
            if (arr[j] > arr[i]) {
                getHigher = true;
                result[i] = j - i;
                break;
            }
        }
        if (!getHigher) result[i] = 0;
    }
    return result;
};

/** 
 * 循环数组中比当前元素大的下一个元素
*/
var nextGreaterElements = function (nums) {
    var result = [];
    for (let i = 0; i < nums.length; i++) {
        let j;
        let back;
        if (i === nums.length - 1) {
            j = 0;
            back = true;
        } else {
            j = i + 1;
            back = false;
        }
        while (nums[j] <= nums[i]) {
            j++;
            if (j === nums.length && !back) {
                back = true;
                j = 0;
            }
        }
        if (j !== nums.length) {
            result[i] = nums[j];
        } else {
            result[i] = -1;
        }
    }
    return result;
};

/** 
 * 把数组中的 0 移到末尾
*/
var moveZeroes = function (nums) {
    for (let i = 0; i < nums.length; i++) {
        if (nums[i] === 0) {
            nums.push(nums.splice(i, 1));
            i--;
        }
    }
    return nums;
};

/**
 * 改造数组
 */
var matrixReshape = function (nums, r, c) {
    if (r * c !== nums.length * (nums[0] || []).length) {
        return nums;
    }
    var row = nums.length;
    var col = nums[0].length;
    var result = [];
    var current = 0;
    while (current < r * c) {
        if (!(current % c)) {
            result.push([]);
        }
        var arr = result[result.length - 1];
        arr.push(nums[Math.floor(current / col)][current % col]);
        current++;
    }
    return result;
};

/** 
 * 找出数组中最长的连续 1
*/
var findMaxConsecutiveOnes = function (nums) {
    var result = 0;
    nums.split('').replace(/01/g, '0,1').replace(/10/g,'1,0').split(',').forEach((item) => {
        if(item[0] === '1' && item.length > result) {
            result = item.length;
        }
    });
    return result;
};
var findMaxConsecutiveOnes = function (nums) {
    let max = 0, curr = 0;
    for (let k of nums) {
        max = Math.max(max, curr += k);
        if (!k) curr = 0;
    }
    return max;
};
/**
 * 有序矩阵查找
 */
var searchMatrix = function (matrix, target) {
    if (!matrix || !matrix.length) return false;
    var rows = matrix.length;
    var cols = (matrix[0] || []).length;
    var row = 0;
    var col = cols - 1;
    while (row >= 0 && col >= 0 && row <= rows - 1 && col <= cols - 1) {
        if (matrix[row][col] === target) {
            return true;
        } else if (matrix[row][col] > target) {
            --col;
        } else {
            ++row;
        }
    }
    return false;
};


/** 
 * 有序矩阵的 Kth Element
 * 这个题看了答案还不会
*/

var arrayNesting = function (nums) {

};

// 五. 链表相关习题
/**
 * 获取两个链表的相交节点
 */
var getIntersectionNode = function (headA, headB) {
    if (!headA || !headB) return null;
    var curA = headA;
    var curB = headB;
    while (curA != curB) {
        // 注释掉的是判断next是否存在，不存在则把指针指向另一个链表开头。这种方法不行，因为假如两个链表没有相交，那么就会出现无限循环
        //  headA = headA.next != null ? headA.next : pHeadB;
        // headB = headB.next != null ? headB.next : pheadA;
        curA = curA == null ? headB : curA.next;
        curB = curB == null ? headA : curB.next;
    }
    return curA;
};

/** 
 * 链表反转
*/
var reverseList = function (head) {
    if(!head || !head.next) return head;
    var pre, next,current;
    current = head;
    while(current) {
        next = current.next;
        current.next = pre;
        pre = current;
        current = next;
    }
    return pre;
};

/** 
 * 合并两个有序链表
 */
function Merge(pHead1, pHead2) {
    if (!pHead1) return pHead2;
    if (!pHead2) return pHead1;
    var current, staticHead;
    if (pHead1.val < pHead2.val) {
        staticHead = current = pHead1;
        pHead1 = pHead1.next;
    } else {
        staticHead = current = pHead2;
        pHead2 = pHead2.next;
    }
    while (pHead1 && pHead2) {
        if (pHead1.val < pHead2.val) {
            current.next = pHead1;
            pHead1 = pHead1.next;
        } else {
            current.next = pHead2;
            pHead2 = pHead2.next;
        }
        current = current.next;
    }
    if (!pHead1) {
        current.next = pHead2;
    }
    if (!pHead2) {
        current.next = pHead1;
    }
    return staticHead
}

/** 
 * 寻找链表中环的入口节点
*/
function EntryNodeOfLoop(phead) {
    if(!phead) return phead;
    var fast = slow = phead;
    while(fast.next && slow) {
        fast = fast.next.next;
        slow = slow.next;
        if(fast === slow) {
            fast = phead;
            while (fast !== slow && fast && slow) {
                fast = fast.next;
                slow = slow.next;
            }
            return slow;
        }
    }
    return null;
}

/** 
 * 删除链表倒数指定位置节点
 * （1）一个指针走完一遍之后从头开始走
 * 注意边界条件和代码执行顺序问题
 */ 
var removeNthFromEnd = function (head, n) {
    if (!head) return head;
    var phead = head;
    var length = 0;
    while (head) {
        length++;
        head = head.next;
    }
    head = phead;
    var count = length - n;
    if(count === 0) {
        return head.next;
    }
    var current = 0;
    while (head && head.next) {
        current++;
        if (count === current) {
            head.next = head.next.next;
        }
        head = head.next;
       
    }
    return phead;
};
/** 
 * 删除链表倒数指定位置节点
 * （2）两个指针：一个先走n步，先走的那个走完之后，另一个正好到达倒数第n的位置
*/
var removeNthFromEnd = function(head, n) {
    var left, before, right = head;
    left = before = {next: head};
    while(n--) {
        if(right) {
            right = right.next;
        } else {
            return head;
        }
    }
    while(right) {
        right = right.next;
        left = left.next;
    }
    left.next = left.next.next;
    return before.next;
}

/** 
 * 交换链表中的相邻节点
*/
var swapPairs = function (head) {
    if(!head || !head.next) return head;
    var phead = head.next;
    var next, pre = {next: head};
    while (head && head.next) {
        next = head.next;
        head.next = next.next;
        next.next = head;
        pre.next = next;

        head = head.next;
        pre = pre.next.next;
    }
    return phead;
};
/** 
 * 链表求和
 */
var addTwoNumbers = function (l1, l2) {

};

/** 
 * 回文链表
 */
var isPalindrome = function (head) {
    const arr = [];
    while(head) {
        arr.push(head.val);
        head = head.next;
    }
    const length = arr.length;
    for (var i = 0; i < Math.floor(length / 2); i++) {
        if (arr[i] !== arr[length - 1 - i]) return false;
    }
    return true;
};

// 六. 树相关习题
// 树的类型：
// 平衡树（左右子树高度差<=1）,
// 二叉查找树（根节点大于等于左子树所有节点，小于等于右子树所有节点） 二叉查找树中序遍历有序。
/** 
 * 求树的高度
*/
var maxDepth = function (root) {
    if(!root) {
        return 0;
    } else {
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }
};
/** 
 * 查看树是否是平衡树
 * (1)理解起来简单，但是有重复计算
*/
var isBalanced = function (root) {
    if(!root) return true;
   var leftHeight = getHeight(root.left); 
   var rightHeight = getHeight(root.right);
    if (Math.abs(leftHeight - rightHeight) > 1) {
       return false;
   } else {
       return isBalanced(root.left) && isBalanced(root.right);
   }
};
// 就是上面的求树的高度
function getHeight(root) {
    if(!root) return 0;
    return Math.max(1 + getHeight(root.left), 1 + getHeight(root.right));
}
/**
 * 查看是否是平衡树
 * (2) 暂定
 */
function isBalanced(root) {
    
}

/** 
 * 树的遍历
 * 1. 广度优先遍历
*/
function levelOrderTraversal(root) {
    if(!root) return [];
    var nodes = [root];
    var values = [];
    while (nodes.length) {
        // 注意这里是shift()，不能是pop()  用pop顺序就不对了
        var node = nodes.shift();
        values.push(node.val);
        node.left && nodes.push(node.left);
        node.right && nodes.push(node.right);
    }
    return values;
}
/**
 * 树的遍历: 
 * 2.1.1 深度优先遍历----前序遍历---递归的方法
*/
function preOrderTraversal(root) {
    if(!root) return [];
    var values = [];
    travel(root, values);
    return values;
}
function travel(root, values) {
    if(!root) return;
    values.push(root.val);
    travel(root.left, values);
    travel(root.right, values);
}
/**
 * 树的遍历:
 * 2.1.2 深度优先遍历----前序遍历---非递归的方法
 * 使用数组模拟栈
*/
function preOrderTraversal(root) {
    if(!root) return [];
    var nodes = [root];
    var values = [];
    while(nodes.length) {
        var node = nodes.pop();
        values.push(node.val);
        // 注意这个地方需要先判断node.right是否存在，存在才能nodes.push(node.right)；否则放进数组中的是null
        node.right && nodes.push(node.right);
        node.left && nodes.push(node.left);
    }
    return values;
}


/**
 * 树的遍历: 递归的方法
 * 2.2.1 深度优先遍历----中序遍历
*/
function midOrderTraversal(root) {
    if(!root) return [];
    var values = [];
    travel(root, values);
}
function travel(root, values) {
    if(!root) return;
    travel(root.left, values);
    values.push(root.val);
    travel(root.right, values);
}

/**
 * 树的遍历:
 * 2.2.2 深度优先遍历----中序遍历---非递归的方法
 * 如果树节点没有重复val可以用这个方法，但是有重复val就不行
 * 因为var index = values.indexOf(node.val)得到的index就可能不是当前节点的val对应的index
*/
function midOrderTraversal(root) {
    if (!root) return [];
    var nodes = [root];
    var values = [root.val];
    while (nodes.length) {
        var node = nodes.pop();
        node.right && nodes.push(node.right);
        node.left && nodes.push(node.left);
        
        var index = values.indexOf(node.val);
        node.left && values.splice(index,0,node.left.val);
        index = values.indexOf(node.val);
        node.right && values.splice(index+1,0,node.right.val);
    }
    return values;
}

/**
 * 树的遍历:
 * 2.2.3 深度优先遍历----中序遍历---非递归方法
 * 有重复值也可以用这方法
 */
function midOrderTraversal(root) {
    if (!root) return [];
    var nodes = [];
    var values = [];
    while (root || nodes.length) {
        if(root) {
            nodes.push(root);
            root = root.left;
        } else {
            root = nodes.pop();
            values.push(root.val);
            root = root.right;
        }
    }
    return values;
}

/**
 * 树的遍历: 递归的方法
 * 2.3.1 深度优先遍历----后序遍历
*/
function postOrderTraversal(root) {
    if (!root) return [];
    var values = [];
    travel(root, values);
}
function travel(root, values) {
    if (!root) return;
    travel(root.left, values);
    travel(root.right, values);
    values.push(root.val);
}
/**
 * 树的遍历
 * 2.3.2 深度优先遍历----后序遍历---非递归的方法
 * 和前序非递归遍历正好相反
*/
function postOrderTraversal(root) {
    if (!root) return [];
    var nodes = [root];
    var values = [];
    while(nodes.length) {
        var node = nodes.pop();
        values.unshift(node.val);
        node.left && nodes.push(node.left);
        node.right && nodes.push(node.right);
    }
    return values;
}

/**
 * 两节点最长路径
 */
function diameterOfBinaryTree(root) {
    var diameter = 0;
    function dfs(root) {
        if(!root) return 0;
        var left = dfs(root.left);
        var right = dfs(root.right);
        diameter = Math.max(diameter, left + right);
        return 1 + Math.max(left, right);
    }
    dfs(root);
    return diameter;
}
/** 
 * 相同节点值的最长路径
*/
function longestUnivaluePath(root) {
    var diameter = 0;
    function dfs(root, val) {
        if(!root) return 0;
        var left = dfs(root.left, root.val);
        var right = dfs(root.right, root.val);
        diameter = Math.max(diameter, left + right);
        return val === root.val ? 1 + Math.max(left, right) : 0;
    }
    dfs(root);
    return diameter;
}

/** 
 * 间隔遍历
*/
function rob(root) {
    if(!root) return 0;
    var val = root.val;
    if(root.left) val += rob(root.left.left) + rob(root.left.right);
    if(root.right) val += rob(root.right.left) + rob(root.right.right);
    var val2 = rob(root.left) + rob(root.right);
    return Math.max(val, val2);
}
/**
 * 找出二叉树中第二小的节点
 */
var findSecondMinimumValue = function (root) {
    if(!root) return 0;
    var min = root.val;
    secondMin = Infinity;
    function dfs(root) {
        if(!root) return;
        if(root.val <= min) {
            min = root.val;
        } else if (root.val < secondMin) {
            secondMin = root.val;
        }
        dfs(root.left);
        dfs(root.right);
    }
    dfs(root);
    return Infinity === secondMin ? -1 : secondMin;
};


/**
 * 翻转树
 * 方法1
 */
function invertTree(root) {
    if(!root) return root;
    var temp = root.right;
    root.right = invertTree(root.left);
    root.left = invertTree(temp);
    return root;
}
/**
 * 翻转树
 * 方法2
 */
var invertTree = function (root) {
    invert(root);
    return root;
};
function invert(root) {
    if (!root) return;
    var temp = root.left;
    root.left = root.right;
    root.right = temp;
    invertTree(root.left);
    invertTree(root.right);
}
/** 
 * 归并两棵树
 */
var mergeTrees = function (t1, t2) {
    if (!t1 && !t2) return null;
    var root = new TreeNode((t1 && t1.val || 0) + (t2 && t2.val || 0));
    root.left = mergeTrees(t1 && t1.left, t2 && t2.left);
    root.right = mergeTrees(t1 && t1.right, t2 && t2.right);
    return root;
};
/** 
 * 有没有一条从根节点到叶子节点路径和等于指定的数字
*/
var hasPathSum = function (root, sum) {
    if (!root) return false;
    return getPathSum(root, 0, sum);
};
function getPathSum(root, currentSum, sum) {
    if (!root.left && !root.right) {
        return currentSum + root.val === sum;
    } else if (root.left && root.right) {
        return getPath(root.left, currentSum + root.val, sum) || getPath(root.right, currentSum + root.val, sum)
    } else if (root.left) {
        getPath(root.left, currentSum + root.val, sum)
    } else {
        return getPath(root.right, currentSum + root.val, sum)
    }
}
/** 
 * 统计路径和等于一个数的路径数量, 不一定要到叶子节点，也不一定是树根节点，但是得是从父节点到子节点
*/
function pathSum(root, sum) {
    let result = 0;
    if (!root) return result;
    result = pathSumStartWithRoot(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    return result;
}
function pathSumStartWithRoot(root, sum) {
    if(!root) return 0;
    var result = 0;
    if (root.val === sum) result++;
    result += pathSumStartWithRoot(root.left, sum - root.val) + pathSumStartWithRoot(root.right, sum - root.val);
    return result;
}

/** 
 * 是否是子树
 * 先判断当前树与目标数是否相等，不相等再看左子树与目标数是否相等，再看右子树与目标数是否相等
*/
var isSubtree = function (s, t) {
    // 这种写法不错，可以学习一下（只要有一个为假，则必须要两个都为假，才能返回真。一个假一个真则会返回假）
    if (!s || !t) return !s && !t;
    return isEqual(s, t) || isSubtree(s.left, t) || isSubtree(s.right, t);
}
function isEqual(s, t) {
    if (!s || !t) return !s && !t;
    if (s.val !== t.val) return false;
    return isEqual(s.left, t.left) && isEqual(s.right, t.right)
}

/** 
 * 树的对称
*/
var isSymmetric = function (root) {
    if (!root) return true;
    return isReversed(root.left, root.right);
};

function isReversed(root1, root2) {
    if (!root1 || !root2) return !root1 && !root2;
    if (root1.val !== root2.val) return false;
    return isReversed(root1.left, root2.right) && isReversed(root1.right, root2.left)
}

/**
 * 树的根节点到叶子节点的最小路径长度(注意和树的高度不一样)
 */
var minDepth = function (root) {
    if (!root) return 0;
    return leafHeight(root, 0);
};
var leafHeight = function (root, sum) {
    if (!root.left && !root.right) return ++sum;
    if (root.left && root.right) return Math.min(leafHeight(root.left, sum + 1), leafHeight(root.right, sum + 1));
    if (root.left) return leafHeight(root.left, sum + 1);
    if (root.right) return leafHeight(root.right, sum + 1);
}

/** 
 * 统计左叶子节点的和
 * 关键点：是否是叶子节点，是否是左节点
*/

function sumOfLeftLeaves(root) {
    if(!root) return 0;
    if (isLeaf(root.left)) {
        return root.left.val + sumOfLeftLeaves(root.right);
    }
    return sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);
}

function isLeaf(root) {
    if(!root) return false;
    return !root.left && !root.right;
}

/** 
 * 一棵树每层节点的平均数
*/
var averageOfLevels = function (root) {
    var levels = {};
    travelLevel(root, 0, levels);
    var resultArr = [];
    for (key in levels) {
        resultArr.push(levels[key].sum / levels[key].count);
    }
    return resultArr;
};
function travelLevel(root, level, levels) {
    if(!root) return;
    levels[level + 1] = levels[level + 1] || {count: 0, sum: 0};
    levels[level + 1].count += 1;
    levels[level + 1].sum += root.val;
    travelLevel(root.left, level + 1, levels);
    travelLevel(root.right, level + 1, levels);
}
/**
 * 得到左下角的节点
 */
var findBottomLeftValue = function (root) {
    var obj = {
        maxLevel: 0,
        value: root.val
    }
    function dfs(root, level) {
        if(!root) return;
        if(level + 1 > obj.maxLevel) {
            obj.maxLevel = level + 1;
            obj.value = root.val;
        }
        if (root.left) dfs(root.left, level + 1);
        if (root.right) dfs(root.right, level + 1);
    }
    dfs(root, 0);
    return obj.value;
};


/** 
 * 修剪二叉查找树
*/
var trimBST = function (root, L, R) {
    if(!root) return root;
    if (root.val > R) return trimBST(root.left, L, R);
    if (root.val < L) return trimBST(root.right, L, R);
    root.left = trimBST(root.left, L, R);
    root.right = trimBST(root.right, L, R);
    return root;
};

/** 
 * 找出二叉查找树中倒数第k小的数
*/
var kthSmallest = function (root, k) {
    var result;
    function dfs(root) {
        if (!root) return;
        dfs(root.left);
        k--;
        if (k === 0) {
            return result = root.val;
        }
        dfs(root.right);
    }
    dfs(root);
    return result
};

/**
 * 把二叉查找树每个节点的值都加上比它大的节点的值
 */
var convertBST = function (root) {
    var sum = 0;
    function dfs(root) {
        if(!root) return;
        dfs(root.right);
        root.val += sum;
        sum = root.val;
        dfs(root.left);
    }
    dfs(root);
    return root;
};

/** 
 * 二叉查找树的最近公共祖先
*/
var lowestCommonAncestor = function (root, p, q) {
    if (root.val > p.val && root.val > q.val) {
        return lowestCommonAncestor(root.left, p, q);
    }
    else if (root.val < p.val && root.val < q.val) {
        return lowestCommonAncestor(root.right, p, q);
    }
    else return root;
};

/** 
 * 二叉树的最近公共祖先
*/
var lowestCommonAncestor = function (root, p, q) {
    if(!root) return root;
    if(hasNode(root.left, p) && hasNode(root.left, q)) {
        return lowestCommonAncestor(root.left, p, q);
    } else if (hasNode(root.right, p) && hasNode(root.right, q)) {
        return lowestCommonAncestor(root.right, p, q);
    } else {
        return root;
    }
};
function hasNode(root, node) {
    if(!root) return false;
    if(root === node) return true;
    return hasNode(root.left, node) || hasNode(root.right, node)
}

// 七. 字符串相关习题
/**
 * 字符串循环移位
 * 左移：从左边第n个切断
 * 右移：从右边第n个切断
 */
function rotateTranslateLeft(str, n){
    var arr = str.split('');
    var left = arr.slice(0, n).reverse();
    var right = arr.slice(n).reverse();
    return left.concat(right).reverse().join('');
}

/**
 * 两个字符串包含的字符是否完全相同
 * 注意 forEach 函数 return 一个值 只是返回了forEach的值，不是返回外层函数的值
 */
var isAnagram = function (s, t) {
    var obj = {};
    var result = true;
    s.split('').forEach((item) => {
        obj[item] = obj[item] ? ++obj[item] : 1;
    });
    t.split('').forEach((item) => {
        if (obj[item]) {
            --obj[item]
        } else {
            result = false;
        }
    });
    for (var i in obj) {
        if (obj[i]) {
            result = false;
        }
    }
    return result;
};

/**
 * 计算一组字符集合可以组成的回文字符串的最大长度
 * 方法1： 把成单的元素数量拎出来
 */
var longestPalindrome = (s) => {
    const count = {};
    for(let value of s ){
        count[value] = (count[value] || 0) + 1;
    }
    let odd = 0;
    for(let key in count) {
        odd += count[key] % 2;
    }
    return s.length - odd + !!odd;
}
/**
 * 计算一组字符集合可以组成的回文字符串的最大长度
 * 方法2： 把成对的元素数量拎出来
 */
var longestPalindrome = (s) => {
    const count = {};
    for (let value of s) {
        count[value] = (count[value] || 0) + 1;
    }
    let sum = 0;
    let hasSingle = false;
    for(let key in count) {
        if (count[key] % 2) {
            hasSingle = true;
            sum += count[key] - 1;
        } else {
            sum += count[key];
        }
    }
    return sum + !!hasSingle;
}

/** 
 * 字符串同构
*/
var isIsomorphic = function (s, t) {
    if(s.length !== t.length) return false;
    var obj_s = {}, obj_t = {};
    for(var i = 0; i < s.length; i++) {
        if (!obj_s[s[i]]) obj_s[s[i]] = t[i];
        if (!obj_t[t[i]]) obj_t[t[i]] = s[i];
        if (obj_s[s[i]] !== t[i] || obj_t[t[i]] !== s[i]) {
            return false;
        }
    }
    return true;
};

/** 
 * 回文子字符串个数
*/
var countSubstrings = function (s) {
    let count = 0;
    for(let i = 0; i<s.length; i++) {
        helper(i, i); // 基数
        helper(i, i + 1); // 偶数
    }
    function helper(low, high) {
        while(low >= 0 && high <= s.length - 1 && s[low] === s[high]) {
            count++;
            low--;
            high++;
        }
    }
    return count;
};

/**
 * 判断一个整数是否是回文数
 * 利用数组
 */
var isPalindrome = function (x) {
    var arr = [];
    var number = x;
    while (number) {
        arr.push(number % 10);
        number = Math.floor(number / 10);
    }
    for (var i = 0; i < Math.ceil(arr.length / 2); i++) {
        if (arr[i] !== arr[arr.length - i - 1]) {
            return false;
        }
    }
    return true;
};
/**
 * 判断一个整数是否是回文数
 * 不使用数组
 */
var isPalindrome = function (x) {
    if (x < 0) return false;
    var originCount = x;
    var num = 0;
    while (x) {
        num = (x % 10) + num * 10;
        x = Math.floor(x / 10);
    }
    return num === originCount;
};
/**
 * 统计二进制字符串中连续 1 和连续 0 数量相同的子字符串个数
 * 自己写的：时间复杂度较高
 */
var countBinarySubstrings = function (s) {
    var result = 0;
    for(var i = 0; i < s.length; i++) {
        result += !!calc(i);
    }
    function valid(number) {
        return ~['0', '1'].indexOf(number);
    }
    function verse(number) {
        if (!valid) return -1;
        return number === '0' ? '1' : '0';
    }
    function calc(start) {
        for(var i = start + 1; i<s.length;) {
            if (s[i] === s[i - 1] && valid(s[i])) {
                i++;
            } else {
                let j = i + (i - start -1);
                while(j > i) {
                    if (s[j] !== s[j - 1] || s[j] === verse(s[start])) {
                        return false;
                    }
                    j--;
                }
                return true;
            }
        }
    }
    return result;
};
/**
 * 统计二进制字符串中连续 1 和连续 0 数量相同的子字符串个数
 * 看别人的：时间复杂度较低
 * 利用正则对字符串进行分类处理
 */
var countBinarySubstrings = (s) => {
    return s.replace(/10/g, '1,0').replace(/01/g, '0,1').split(',').reduce((res, item, index, arr) => {
        return index ? res += Math.max(item.length, arr[--index].length) : 0;
    }, 0);
}

// 八. 算法思想
// (1) 双指针
/**
 *  两数平方和
 */
var judgeSquareSum = function (c) {
    var i = 0; 
    var j = Math.floor(Math.sqrt(c));
    while(i <= j) {
        var result = Math.pow(i, 2) + Math.pow(j, 2);
        if (result === c) {
            return true;
        } else if(result > c) {
            j--;
        } else {
            i++;
        }
    }
    return false;
};

/**
 * 反转字符串中的元音字符
 */
var reverseVowels = function (s) {
    var yuanyin = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'];
    var i = 0;
    var j = s.length - 1;
    var result = [];
    // 注意这里要有等于号，因为当数组只有一个元素的时候i和j是相等的
    while(i <= j) {
        if (~yuanyin.indexOf(s[i]) && ~yuanyin.indexOf(s[j])) {
            result[i] = s[j];
            result[j] = s[i];
            i++;
            j--;
        } else {
            if (~yuanyin.indexOf(s[i])) {
                result[j] = s[j];
                j--;
            } else if (~yuanyin.indexOf(s[j])) {
                result[i] = s[i];
                i++;
            } else {
                result[i] = s[i];
                result[j] = s[j];
                i++;
                j--;
            }
        }
    }
    return result.join('');
};

/**
 * 回文字符串
 * 可以删除一个字符，判断是否能构成回文字符串。
 * 自己写的：性能不行(构造了很多数组和字符串)
 * 别人写的只是在原来的字符串上进行比较，没有构造任何多余的数据结构
 */
var validPalindrome = function (s, deleteCount = 1) {
    var i = 0;
    var j = s.length - 1;
    while (i <= j) {
        if (s[i] === s[j]) {
            j--;
            i++;
        } else {
            const arr1 = s.split('').concat([]);
            arr1.splice(i, 1);
            const arr2 = s.split('').concat([]);
            arr2.splice(j, 1);
            return deleteCount ? (validPalindrome(arr1.join(''), 0) || validPalindrome(arr2.join(''), 0)) : false;
        }
    }
    return true;
};
/**
 * 回文字符串
 * 可以删除一个字符，判断是否能构成回文字符串。
 * 看别人的
 */
var validPalindrome = (s) => {
    var i = 0;
    var j = s.length - 1;
    while(i <= j) {
        if(s[i] !== s[j]) {
            return isPalindrome(s, i + 1, j) || isPalindrome(s, i, j-1);
        }
        i++;
        j--;
    }
    return true;
}
function isPalindrome(s, i, j) {
    while(i <= j) {
        if(s[i] !== s[j]) {
            return false;
        }
        i++;
        j--;
    }
    return true;
}

// (2)贪心思想
/** 
 * 分配饼干
*/
var findContentChildren = function (g, s) {
    var content = 0;
    g.sort((a, b) => {
        return a-b;
    });
    s.sort((a,b) => {
        return a-b;
    });
    var i = 0, j = 0;
    while(i < g.length && j < s.length) {
       if(s[j] >= g[i]) {
            content++;
            i++;
            j++;
        } else {
            j++;
        }
    }
    return content;
};
/** 
 * 不重叠的区间个数
*/
var eraseOverlapIntervals = function (intervals) {
    if (!intervals.length || !intervals[0].length) return 0;
    intervals.sort((a, b) => a[1] - b[1]);
    var count = 0;
    intervals.reduce((pre, item) => {
        if (pre[1] > item[0]) {
            count++;
            return pre;
        } else {
            return item;
        }
    });
    return count;
};

/** 
 * 投飞镖刺破气球
 * 自己写的：有的case没通过
*/
var findMinArrowShots = function (s) {
    if (!s.length || !s[0].length) return 0;
    s.sort((a, b) => a[0] - b[0]);
    var count = 0;
    while (s.length) {
        var arr = [];
        for (var i = 1; i <= s.length - 1; i++) {
            const inter = intersaction(s[0], s[i]);
            let hasTheSameInter = false;
            if (inter.length > 0) {
                for (let i = 0; i < arr.length; i++) {
                    if (intersaction(arr[i], inter).length > 0) {
                        hasTheSameInter = true;
                        arr[i] = intersaction(arr[i], inter);
                        break;
                    }
                }
                if (!hasTheSameInter) arr.push(inter);
                s.splice(i, 1);
                i--;
            }
        }
        count += (arr.length || 1);
        s.splice(0, 1);
    }
    return count;
};
function intersaction(arr1, arr2) {
    if (arr1[1] < arr2[0] || arr2[1] < arr1[0]) return [];
    if (arr1[0] <= arr2[0]) {
        return [arr2[0], Math.min(arr1[1], arr2[1])];
    } else {
        return [arr1[0], Math.min(arr1[1], arr2[1])]
    }
}
/**
 * 投飞镖刺破气球
 * 此方法通过尾数排序，能确定只要有交叉就能一剑刺破
 * 上面我自己的方法是通过首数排序，这样排序后交叉也不能代表就一定能一剑刺破。
*/
var findMinArrowShots = function (s) {
    if (!s.length || !s[0].length) return 0;
    s.sort((a, b) => a[1] - b[1]);
    var count = 0;
    while (s.length > 0) {
        for (var i = 1; i < s.length; i++) {
            if (s[0][1] >= s[i][0]) {
                s.splice(i, 1);
                i--;
            }
        }
        s.splice(0, 1);
        count++;
    }
    return count;
}

/**
 * 根据身高和序号重组队列
 */
var reconstructQueue = function (people) {
    if (!people.length || !people[0].length || people.length == 1) return people;
    people.sort((a, b) => b[0] === a[0] ? a[1] - b[1] : b[0] - a[0]);
    var result = people.reduce((pre, item) => {
        pre.splice(item[1], 0, item);
        return pre;
    }, []);
    return result;
};

/** 
 * 种植花朵
*/
var canPlaceFlowers = function (flowerbed, n) {
    var count = 0;
    for (var i = 0; i < flowerbed.length; i++) {
        if (flowerbed[i] === 1) continue;
        var pre = i > 0 ? flowerbed[i - 1] : 0;
        var next = i < flowerbed.length - 1 ? flowerbed[i + 1] : 0;
        if (!pre && !next) {
            flowerbed[i] = 1;
            count++;
        }
    }
    return count >= n;
};

/**
 * 判断是否为子序列
 * s = "abc", t = "ahbgdc" ==> true
 * s = "axc", t = "ahbgdc" ===> false
 * isSubsequence1是我自己的写的，利用的是indexOf。如果t里面有重复字符就会出问题。
 * isSubsequence是别人的，不管是否重复都没问题。因为t.substr直接去掉了已经匹配的部分了
 */
var isSubsequence1 = function (s, t) {
    var preIndex = 0;
    for (let i = 0; i < s.length; i++) {
        const currentIndex = t.indexOf(s[i]);
        if (currentIndex < preIndex) return false;
        preIndex = currentIndex;
    }
    return true;
};

var isSubsequence = function (s, t) {
    if (s.length === 0) {
        return true;
    }
    if (s.length !== 0 && t.length === 0) {
        return false;
    }
    for (var i = 0; i < s.length; i++) {
        var temp = t.indexOf(s[i]);
        if (temp < 0) {
            return false;
        } else {
            t = t.substr(temp + 1);
        }
    }
    return true;
};
/** 
 * 修改一个数成为非递减数组
*/
var checkPossibility = function (nums) {
    var count = 0;
    for (let i = 1; i < nums.length && count < 2; i++) {
        if (nums[i] >= nums[i - 1]) continue;
        count++;
        if (i - 2 >= 0 && nums[i - 2] > nums[i]) {
            nums[i] = nums[i - 1];
        } else {
            nums[i - 1] = nums[i];
        }
    }
    return count < 2;
};

// (3) 动态规划
// 动态规划有三个核心元素：
// 1.最优子结构
// 2.边界
// 3.状态转移方程

// 什么样的问题适用DP
// 1) 问题是由交叠的子问题所构成，大问题分解为小问题。
// 2) 将交叠子问题的一次次求解→子问题只求解一次，并将结果记录保存。
// 3) 利用空间(子问题存储)来换取时间

/** 斐波那契数列之爬楼梯
 * 递归解法：时间复杂度O(2^n)
*/
var climbStairs = function (n) {
    if(n === 0) {
        return 0;
    }
    if(n === 1) return 1;
    if(n === 2) return 2;
    return climbStairs(n-2) + climbStairs(n - 1);
};
/**
 * 爬楼梯
 * 递归解法: 时间复杂度O(n),空间复杂度O(1)
 * dp[i] = dp[i-2] + dp[i-1]
*/
var climbStairs = function (s) {
    if(s <= 2) return s;
    var pre1 = 1, pre2 = 2;
    var current;
    for(i = 3; i<= s; i++) {
        current = pre1 + pre2;
        pre1 = pre2;
        pre2 = current;
    }
    return current;
}
/**
 * 抢劫一排住户，但是不能抢邻近的住户，求最大抢劫量
 * 自己写的：复杂度太高（和上面树抢劫一个做法）
 */
function rob(nums) {
    var result = rob(nums, 0);
    return result;
}
var _rob = function (nums, start) {
    if(start > nums.length - 1) return 0;
    var val = nums[start];
    var val2 = 0;
    if(start + 2 < nums.length) {
        val = val + _rob(nums, start + 2)
    }
    if(start + 1 < nums.length) {
        val2 = _rob(nums, start + 1);
    }
    return Math.max(val, val2);
};
/**
 * 抢劫一排住户，但是不能抢邻近的住户，求最大抢劫量
 * 看别人的
 * dp[i] = Max(dp[i-2] + nums[i], dp[i-1])
 * 用一个数组存下之前的数据
 */
var rob = function (nums) {
    if (nums.length < 2) return nums;
    var pre1 = nums[0], pre2 = Math.max(nums[1], nums[0]);
    var dp = [pre1, pre2];
    for(var i = 2; i <= nums.length - 1; i++) {
        dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i]);
    }
    return dp[dp.length - 1];
}
/**
 * 抢劫一排住户，但是不能抢邻近的住户，求最大抢劫量
 * 看别人的
 * dp[i] = Max(dp[i-2] + nums[i], dp[i-1])
 * 用两个变量存下之前的数据
 */
var rob = function (nums) {
    if(nums.length < 2) return nums;
    var pre1 = nums[0], pre2 = Math.max(nums[0], nums[1]);
    var current = pre2;
    for(var i = 2; i <= nums.length - 1; i++) {
        current = Math.max(pre1 + nums[i], pre2);
        pre1 = pre2;
        pre2 = current;
    }
    return current;
}
/** 
 * 强盗在环形街区抢劫
*/
var circleRob = function (num) {
    if(num.length <= 1) return num;
    return Math.max(_rob(num, 0, num.length - 2), _rob(num, 1, num.length - 1));
}
var _rob = function(nums, start, end) {
    if (end - start <= 1) return nums[start];
    var pre1 = nums[start], pre2 = Math.max(nums[start], nums[start + 1]);
    var current = pre2;
    for (var i = start + 2; i <= end; i++) {
        current = Math.max(pre1 + nums[i], pre2);
        pre1 = pre2;
        pre2 = current;
    }
    return current;
}

/** 
 * 矩阵的最小路径和
*/
var minPathSum = function (grid) {
    if (!grid.length || !grid[0].length) return 0;
    var row = grid.length;
    var col = grid[0].length;
    var dp = [];
    for (var i = 0; i <= row - 1; i++) {
        dp.push([grid[0][0]]);
        for (var j = 0; j <= col - 1; j++) {
            if (i === 0) {
                dp[0][j] = j > 0 ? dp[0][j - 1] : 0;
            } else if (j === 0) {
                dp[i][0] = i > 0 ? dp[i - 1][0] : dp[i][0];
            } else {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]);
            }
            dp[i][j] = dp[i][j] + grid[i][j];
        }
    }
    return dp[row - 1][col - 1];
};

/** 
 * 矩阵的总路径数
*/
var uniquePaths = function(m,n) {
    var dp = new Array(m).fill(new Array(n).fill(1));
    for(let i = 1; i < m; i++) {
        for(let j = 1; j < n; j++) {
            dp[i][j] =  dp[i][j-1] + dp[i-1][j];
        }
    }
    return dp[m-1][n-1];
}

// 动态规划 之  背包客


