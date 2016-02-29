import java.util.*;


class ListNode {
    int val;
    ListNode next;
    ListNode(int val) {
        this.val = val;
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int val) {
        this.val = val;
    }
}

class TreeLinkNode {
    TreeLinkNode left;
    TreeLinkNode right;
    TreeLinkNode next;
}

public class Solution {

    /*
     *  Subsets
     *  Given a set of distinct integers, nums, return all possible subsets.
     *
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> results = new LinkedList<List<Integer>>();
        results.add(new LinkedList<Integer>());
        Arrays.sort(nums);
        int size = results.size();
        for (int i=0; i<nums.length; i++) {
            for (int j=0; j<size; j++) {
                List<Integer> temp = new LinkedList<Integer>(results.get(j));
                temp.add(nums[i]);
                results.add(temp);
            }
            size = results.size();
        }
        return results;
    }


    /*
    *  Subsets II
    *  Given a collection of integers that might contain duplicates, nums, return all possible subsets.
    *
    */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        result.add(new LinkedList<Integer>());
        Arrays.sort(nums);
        int start = 0;
        int size = result.size();
        for (int i=0; i<nums.length; i++) {
            for (int j=start; j<size; j++) {
                LinkedList<Integer> temp = new LinkedList<Integer>(result.get(j));
                temp.add(nums[i]);
                result.add(temp);
            }

            if (i<nums.length - 1 && nums[i] == nums[i+1]) {
                start = size;
            } else {
                start = 0;
            }
            size = result.size();
        }
        return result;
    }



    public String ZigZagconvert(String s, int nRows) {
        if (s == null || s.length() == 0 || nRows < 0) {
            return "";
        }

        if (nRows == 1) {
            return s;
        }
        StringBuilder result = new StringBuilder();
        int size = 2*nRows-2;
        for (int i=0; i<nRows; i++) {
            for (int j=i; j<s.length(); j+=size) {
                result.append(s.charAt(j));
                if (i==0 && i== nRows-1) {
                    int index = j+size-2*i;
                    if (index < s.length()) {
                        result.append(s.charAt(index));
                    }
                }
            }
        }
        return result.toString();
    }

    /*
     *  Find the Duplicate Number
     *  Given an array nums containing n + 1 integers where each integer is between 1 and n
     *
     */

    public int findDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        int result = 0;
        for (int i=0; i<nums.length; i++) {
            if (!set.contains(nums[i])) {
                set.add(nums[i]);
            } else {
                result = nums[i];
                return result;
            }
        }
        return result;
    }

    /*
     *  Longest Palindromic Substring
     *  unique longest palindromic substring.
     *
     */
    public String longestPalindrome(String s) {
        String result = "";
        for (int i=0; i<s.length(); i++) {
            String temp = helper(s, i, i);
            if (temp.length() > result.length()) {
                result = temp;
            }
            temp = helper(s, i, i+1);
            if (temp.length() > result.length()) {
                result = temp;
            }
        }
        return result;
    }


    public String helper(String s, int begin, int end) {
        while (begin >= 0 && end < s.length() && s.charAt(begin) == s.charAt(end)) {
            begin--;
            end++;
        }
        return s.substring(begin + 1, end);
    }

    /*
     *  Letter Combinations of a Phone Number
     *  Input:Digit string "23"
     *  Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
     */
    public List<String> letterCombinations(String digits) {
        String[] map = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

        List<String> result = new LinkedList<String>();
        if (digits == null || digits.length() == 0) {
            return result;
        }
        rec(result, "", digits, map);
        return result;
    }

    public void rec(List<String> result, String temp, String digits, String[] map) {
        if (temp.length() == digits.length()) {
            result.add(temp);
        } else {
            String options = map[digits.charAt(temp.length()) - '0'];
            for (int i=0; i<options.length(); i++) {
                rec(result, temp+String.valueOf(options.charAt(i)), digits, map);
            }
        }
    }

    /*
     *  4Sum
     *  Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target?
     *  Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
     */

    public List<List<Integer>> fourSum(int[] nums, int target) {
        HashSet<ArrayList<Integer>> hashSet = new HashSet<ArrayList<Integer>>();
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        for (int i = 0; i <= nums.length-4; i++) {
            for (int j = i + 1; j <= nums.length-3; j++) {
                int low = j + 1;
                int high = nums.length - 1;

                while (low < high) {
                    int sum = nums[i] + nums[j] + nums[low] + nums[high];

                    if (sum > target) {
                        high--;
                    } else if (sum < target) {
                        low++;
                    } else if (sum == target) {
                        ArrayList<Integer> temp = new ArrayList<Integer>();
                        temp.add(nums[i]);
                        temp.add(nums[j]);
                        temp.add(nums[low]);
                        temp.add(nums[high]);

                        if (!hashSet.contains(temp)) {
                            hashSet.add(temp);
                            result.add(temp);
                        }

                        low++;
                        high--;
                    }
                }
            }
        }
        return result;
    }


    /*
     *
     *   Two Sum
     *   Given an array of integers, find two numbers such that they add up to a specific target number.
     *   Input: numbers={2, 7, 11, 15}, target=9
     *   Output: index1=1, index2=2
     */

    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int i=0; i<nums.length; i++) {
            if (map.containsKey(target-nums[i])) {
                return new int[] {map.get(target-nums[i])+1, i+1};
            } else {
                map.put (nums[i], i);
            }
        }
        throw new RuntimeException();
    }
    //if the array is sorted in ascending
    public int[] twoSum2(int[] nums, int target) {
        Arrays.sort(nums);
        int i = 0, j = nums.length - 1;
        while (i < j) {
            int sum = nums[i] + nums[j];
            if (sum > target) {
                j--;
            } else if (sum < target) {
                i++;
            } else {
                return new int[]{i+1, j+1};
            }
        }
        throw new RuntimeException();
    }


    /*
     *  3Sum
     *  Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of
     *  zero
     */

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        for (int i=0; i<nums.length; i++) {
            if (nums[i] > 0) {
                break;
            }
            for (int j=i+1; j<nums.length; j++) {
                if (nums[i] + nums[j] > 0 && nums[j] > 0) {
                    break;
                }

                for (int k=j+1; k<nums.length; k++) {
                    if (nums[i] + nums[j] + nums[k] == 0) {
                        List<Integer> temp = new LinkedList<Integer>();
                        temp.add(nums[i]);
                        temp.add(nums[j]);
                        temp.add(nums[k]);
                        result.add(temp);
                    }
                    while(k+1<nums.length && nums[k] == nums[k+1]) {
                        k++;
                    }
                }
                while(j+1<nums.length && nums[j] == nums[j+1]) {
                    j++;
                }
            }
            while(i+1<nums.length && nums[i] == nums[i+1]) {
                i++;
            }
        }

        return result;
    }

    /*
     *  Next Permutation
     *  1,2,3 → 1,3,2
     *  3,2,1 → 1,2,3
     *  1,1,5 → 1,5,1
     *
     */
    public void nextPermutation(int[] nums) {
        int i = 0, j = 0;
        for (i = nums.length - 2; i >= 0; i--) {
            if (nums[i] >= nums[i+1]) {
                continue;
            }
            for (j=nums.length - 1; j>i; j--) {
                if (nums[j] > nums[i]) {
                    break;
                }
            }
            break;
        }
        if (i >= 0) {
            int temp = nums[j];
            nums[j] = nums[i];
            nums[i] = temp;
        }

        int start = i + 1;
        int end = nums.length - 1;
        while (start < end) {
            int tmp = nums[start];
            nums[start] = nums[end];
            nums[end] = tmp;
            start++;
            end--;
        }
    }

    /*
     *  Reverse Words in a String
     *  Given an input string, reverse the string word by word.
     *  Given s = "the sky is blue", return "blue is sky the".
     */
    public String reverseWords(String s) {
        String[] arr = s.split(" ");
        StringBuilder sb = new StringBuilder();
        for (int i=arr.length-1; i>=0; i--) {
            if (!arr[i].equals("")) {
                sb.append(arr[i]);
                sb.append(" ");
            }
        }
        return sb.length() == 0? "" : sb.substring(0, sb.length()-1);
    }

    /*
     *  Single Number
     *  Given an array of integers, every element appears twice except for one. Find that single one.
     *
     */
    public int singleNumber(int[] nums) {
        int result = 0;
        for(int num: nums) {
            result = result ^ num;
        }

        return result;
    }

    public int singleNumber2(int[] nums) {
        int ones = 0, twos = 0, threes = 0;
        for (int i = 0; i < nums.length; i++) {
            twos |= ones & nums[i];
            ones ^= nums[i];
            threes = ones & twos;
            ones &= ~threes;
            twos &= ~threes;
        }
        return ones;
    }

    /*
     *  Compare Version Numbers
     *  0.1 < 1.1 < 1.2 < 13.37
     *
     */
    public int compareVersion(String version1, String version2) {
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");
        int i = 0;
        while (i < Math.max(v1.length, v2.length)) {
            String value1 = i<v1.length? v1[i] : "0";
            String value2 = i<v2.length? v2[i] : "0";
            if (Integer.parseInt(value1) == Integer.parseInt(value2)) {
                i++;
            } else {
                return Integer.parseInt(value1) > Integer.parseInt(value2)? 1 : -1;
            }

        }
        return 0;
    }


    /*
     *  Number of 1 Bits
     *  For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.
     *
     */
    public int hammingWeight(int n) {
        int count = 0;
        while(n != 0) {
            n = n & (n-1);
            count++;
        }
        return count;
    }

    /*
    *
    * Partition List
    * Given a linked list and a value x, partition it such that all
    * nodes less than x come before nodes greater than or equal to x.
    * For example,
    * Given 1->4->3->2->5->2 and x = 3,
    * return 1->2->2->4->3->5.
    *
    */
    public ListNode partition(ListNode head, int x) {
        if (head == null) {
            return head;
        }
        ListNode helper1 = new ListNode(-1);
        ListNode helper2 = new ListNode(-1);
        ListNode temp = head;
        ListNode pre1 = helper1;
        ListNode pre2 = helper2;
        while (temp != null) {
            if (temp.val < x) {
                pre1.next = temp;
                pre1 = pre1.next;
                temp = temp.next;
            } else {
                pre2.next = temp;
                pre2 = pre2.next;
                temp = temp.next;
            }
        }
        pre2.next = null;
        pre1.next = helper2.next;

        return helper1.next;
    }


    /*
     *  Combinations
     *  Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
     *  For example,
     *  If n = 4 and k = 2, a solution is:
     *  [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
     *
     *
     */

    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        List<Integer> al = new LinkedList<Integer>();
        rec(n, k, 1, al, result);
        return result;
    }

    public void rec(int n, int k, int pos, List<Integer> al, List<List<Integer>> result) {
        if(k == 0) {
            result.add(new LinkedList<Integer>(al));
            return;
        }

        for(int i=pos; i<=n-k+1; i++) {
            al.add(i);
            rec(n, k-1, i+1, al, result);
            al.remove(al.size()-1);
        }
    }


    /*
     *
     *
     *  Valid Anagram
     *  Given two strings s and t, write a function to determine if t is an anagram of s.
     *  s = "anagram", t = "nagaram", return true.
     *  s = "rat", t = "car", return false.
     *
     */

    public boolean isAnagram(String s, String t) {
        char[] sArr = s.toCharArray();
        char[] tArr = t.toCharArray();

        Arrays.sort(sArr);
        Arrays.sort(tArr);

        return String.valueOf(sArr).equals(String.valueOf(tArr));
    }


    /*
     *  Unique Paths
     *  A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
     *
     *  m and n will be at most 100.
     *
     */

    public int uniquePaths(int m, int n) {
        if(m==0 || n==0) {
            return 0;
        }
        if(m==1 || n==1) {
            return 1;
        }

        int[][] dp = new int[m][n];

        //left column
        for(int i=0; i<m; i++) {
            dp[i][0] = 1;
        }

        //top row
        for(int j=0; j<n; j++) {
            dp[0][j] = 1;
        }

        //fill up the dp table
        for(int i=1; i<m; i++) {
            for(int j=1; j<n; j++) {
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }

        return dp[m-1][n-1];
    }


    /*
     *  Unique Paths II
     *  Now consider if some obstacles are added to the grids. How many unique paths would there be?
     *  An obstacle and empty space is marked as 1 and 0 respectively in the grid.
     *
     */

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if(obstacleGrid==null||obstacleGrid.length==0) {
            return 0;
        }


        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;

        if(obstacleGrid[0][0]==1||obstacleGrid[m-1][n-1]==1) {
            return 0;
        }



        int[][] dp = new int[m][n];
        dp[0][0]=1;

        //left column
        for(int i=1; i<m; i++) {
            if(obstacleGrid[i][0]==1) {
                dp[i][0] = 0;
            } else {
                dp[i][0] = dp[i-1][0];
            }
        }

        //top row
        for(int i=1; i<n; i++) {
            if(obstacleGrid[0][i]==1) {
                dp[0][i] = 0;
            }else{
                dp[0][i] = dp[0][i-1];
            }
        }

        //fill up cells inside
        for(int i=1; i<m; i++) {
            for(int j=1; j<n; j++) {
                if(obstacleGrid[i][j]==1) {
                    dp[i][j]=0;
                } else {
                    dp[i][j]=dp[i-1][j]+dp[i][j-1];
                }

            }
        }

        return dp[m-1][n-1];
    }

    /*
     *  Path Sum
     *
     *  Given a binary tree and a sum, determine if the tree has a root-to-leaf path such
     *  that adding up all the values along the path equals the given sum.
     *
     *
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null) {
            return false;
        }

        if(root.val == sum && root.left == null && root.right == null) {
            return true;
        }

        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }

    public boolean hasPathSum1(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        boolean[] result = new boolean[]{false};
        rec(root, result, sum, 0);
        return result[0];
    }

    public void rec(TreeNode node, boolean[] result, int sum, int temp) {
        if (node.left == null && node.right == null) {
            if (temp + node.val == sum) {
                result[0] = true;
            }
        } else {
            if (node.left != null) {
                rec(node.left, result, sum, temp+node.val);
            }

            if (node.right != null) {
                rec(node.right, result, sum, temp+node.val);
            }
        }
    }

    /*
     *  Set Matrix Zeroes
     *
     *  Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.
     *
     *
     */

    public void setZeroes(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;

        if(m==0 || n==0) {
            return;
        }

        int[] flagr = new int[m];
        int[] flagc = new int[n];

        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                if(matrix[i][j]==0) {
                    flagr[i]= 1;
                    flagc[j]= 1;
                }
            }
        }

        for(int i=0;i<m;i++) {
            for(int j=0;j<n;j++) {
                if(flagr[i] == 1 || flagc[j] == 1){
                    matrix[i][j]=0;
                }
            }
        }
    }

    /*
     *
     *   Pascal's Triangle
     *   Given numRows, generate the first numRows of Pascal's triangle.
     *   For example, given numRows = 5,
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        if (numRows <= 0) {
            return result;
        }
        List<Integer> pre = new LinkedList<Integer>();
        pre.add(1);
        result.add(pre);
        for (int i = 2; i <= numRows; i++) {
            List<Integer> curr = new LinkedList<Integer>();
            curr.add(1);
            for (int j=0; j<pre.size() - 1; j++) {
                curr.add(pre.get(j) + pre.get(j+1));
            }
            curr.add(1);
            result.add(curr);
            pre = curr;
        }

        return result;
    }


    /*
     *
     *   Pascal's Triangle II
     *   Given numRows, generate the first numRows of Pascal's triangle.
     *   For example, given numRows = 5,
     */
    public List<Integer> getRow(int rowIndex) {
        List<Integer> result = new LinkedList<Integer>();
        result.add(1);
        for (int i=1; i<=rowIndex; i++) {
            for (int j=i-1; j>=1; j--) {
                int temp = result.get(j-1) + result.get(j);
                result.set(j, temp);
            }
            result.add(1);
        }
        return result;
    }


    /*
     *  Search Insert Position
     *
     *  [1,3,5,6], 5 → 2
     *  [1,3,5,6], 2 → 1
     *  [1,3,5,6], 7 → 4
     *  [1,3,5,6], 0 → 0
     */
    public int searchInsert(int[] nums, int target) {
        if(nums[0] >= target) {
            return 0;
        }

        for (int i=0; i<nums.length-1; i++) {
            if(nums[i] < target && nums[i+1] >= target) {
                return i+1;
            }
        }

        return nums.length;
    }


    /*
     *
     *   Validate Binary Search Tree
     *   Given a binary tree, determine if it is a valid binary search tree (BST).
     *
     *
     *
     */

    public boolean isValidBST(TreeNode root) {
        return isValidBST(root, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
    }

    public boolean isValidBST(TreeNode p, double min, double max) {
        if(p == null) {
            return true;
        }

        if(p.val <= min || p.val >= max) {
            return false;
        }

        return isValidBST(p.left, min, p.val) && isValidBST(p.right, p.val, max);
    }


    /*
     *
     *   Unique Binary Search Trees
     *   Given n, how many structurally unique BST's (binary search trees) that store values 1...n?
     *
     */

    public int numTrees(int n) {
        int[] count = new int[n + 1];

        count[0] = 1;
        count[1] = 1;

        for (int i = 2; i <= n; i++) {
            for (int j = 0; j <= i - 1; j++) {
                count[i] = count[i] + count[j] * count[i - j - 1];
            }
        }

        return count[n];
    }


    /*
     *
     *   Unique Binary Search Trees II
     *   Given n, generate all structurally unique BST's (binary search trees) that store values 1...n.
     *
     */

    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new LinkedList<TreeNode>();
        }
        return generateTrees(1, n);
    }

    public List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> list = new LinkedList<TreeNode>();

        if (start > end) {
            list.add(null);
            return list;
        }

        for (int i = start; i <= end; i++) {
            List<TreeNode> lefts = generateTrees(start, i - 1);
            List<TreeNode> rights = generateTrees(i + 1, end);
            for (TreeNode left : lefts) {
                for (TreeNode right : rights) {
                    TreeNode node = new TreeNode(i);
                    node.left = left;
                    node.right = right;
                    list.add(node);
                }
            }
        }

        return list;
    }


    /*
     *
     *   Symmetric Tree
     *   Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
     *
     */

    public boolean isSymmetric(TreeNode root) {
        if(root == null) {
            return true;
        } else {
            return isSymmetric(root.left, root.right);
        }
    }

    public boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        } else if (left == null || right == null) {
            return false;
        } else if (left.val == right.val) {
            return isSymmetric(left.left, right.right) && isSymmetric(left.right, right.left);
        } else {
            return false;
        }
    }


    /*
     *
     *   Sum Root to Leaf Numbers
     *   Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
     *
     */

    public int sumNumbers(TreeNode root) {
        if(root == null) {
            return 0;
        }


        return dfs(root, 0, 0);
    }

    public int dfs(TreeNode node, int num, int sum) {
        if(node == null) {
            return sum;
        }

        num = num*10 + node.val;

        // leaf
        if(node.left == null && node.right == null) {
            sum += num;
            return sum;
        }

        // left subtree + right subtree
        return dfs(node.left, num, sum) + dfs(node.right, num, sum);

    }


    public int sumNumbers1(TreeNode root) {
        if(root == null) {
            return 0;
        }
        int[] sum = new int[]{0};
        rec (root, 0, sum);
        return sum[0];

    }

    public void rec(TreeNode node, int temp, int[] sum) {
        if (node.left == null && node.right == null) {
            int k = temp*10 + node.val;
            sum[0] += k;
        } else {
            temp = temp*10 + node.val;
            if (node.left != null) {
                rec(node.left, temp, sum);
            }

            if (node.right != null) {
                rec(node.right, temp, sum);
            }
        }
    }


    /*
     *
     *   Same Tree
     *   Given two binary trees, write a function to check if they are equal or not.
     *
     */

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        } else if (p == null || q == null) {
            return false;
        } else {
            if (p.val == q.val) {
                return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
            } else {
                return false;
            }

        }
    }

    /*
     *
     *   Move Zeroes
     *   For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].
     *
     */
    public void moveZeroes(int[] nums) {
        int i = 0;
        int j = 0;
        int count = 0;
        while (j<nums.length) {
            if (nums[j] != 0) {
                nums[i] = nums[j];
                i++;
            } else {
                count++;
            }
            j++;
        }

        int k=nums.length-count;
        while (k < nums.length) {
            nums[k] = 0;
            k++;
        }
    }


    /*
     *
     *   Excel Sheet Column Number
     *   Given a column title as appear in an Excel sheet, return its corresponding column number.
     *   AA -> 27 AB -> 28
     */
    public int titleToNumber(String s) {
        int sum = 0;
        int t = 0;
        for (int i=s.length() - 1; i >= 0; i--) {
            sum += Math.pow(26, t) * (s.charAt(i) - 'A' + 1);
            t++;
        }
        return sum;
    }


    /*
     *
     *   Excel Sheet Column Title
     *   Given a positive integer, return its corresponding column title as appear in an Excel sheet.
     *   27 -> AA
     */

    public String convertToTitle(int n) {
        StringBuilder sb = new StringBuilder();
        while(n > 0) {
            n--;
            char tmp = (char)(n%26 + 'A');
            sb.append(tmp);
            n = n/ 26;
        }
        sb.reverse();
        return sb.toString();
    }


    /*
     *
     *   Factorial Trailing Zeroes
     *   Given an integer n, return the number of trailing zeroes in n!.
     *
     */

    public int trailingZeroes(int n) {
        int count = 0;
        for (long i = 5; n/i >= 1; i = i*5) {
            count += n/i;
        }
        return count;
    }


    /*
     *
     *   Rotate Array
     *   For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].
     *
     */
    public void rotate(int[] nums, int k) {
        int len = nums.length;
        int [] temp = new int[len];
        for (int i=0; i<len; i++) {
            temp[i] = nums[i];
        }

        for (int i=0; i<len; i++) {
            nums[(i+k)%len] = temp[i];
        }
    }


    /*
     *
     *   Find Minimum in Rotated Sorted Array
     *   Suppose a sorted array is rotated at some pivot unknown to you beforehand.
     *   (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
     */
    public int findMin(int[] nums) {
        int start=0, end=nums.length-1;

        while (start<end) {
            if (nums[start] <= nums[end])
                return nums[start];

            int mid = (start+end)/2;

            if (nums[mid] >= nums[start]) {
                start = mid+1;
            } else {
                end = mid;
            }
        }

        return nums[start];
    }

    /*
     *
     *   Generate Parentheses
     *   Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
     *   For example, given n = 3, a solution set is:
     *   "((()))", "(()())", "(())()", "()(())", "()()()"
     */

    public List<String> generateParenthesis(int n) {
        List<String> result = new LinkedList<String>();
        dfs(result, "", n, n);
        return result;
    }

    public void dfs(List<String> result, String temp, int left, int right) {
        if(left > right) {
            return ;
        } else if (left == 0 && right == 0) {
            result.add(temp);
        } else {
            if (left > 0) {
                dfs(result, temp+"(", left-1, right);
            }

            if (right > 0) {
                dfs(result, temp+")", left, right-1);
            }
        }
    }


    /*
     *
     *   Gray Code
     *   For example, given n = 2, return [0,1,3,2]. Its gray code sequence is
     *
     */
    public List<Integer> grayCode(int n) {
        List<Integer> result = new LinkedList<Integer>();
        result.add(0);
        for(int i=0; i<n; i++) {
            int inc = 1 << i;
            for(int j=result.size() - 1; j >= 0 ; j--) {
                result.add(result.get(j) + inc);
            }
        }

        return result;
    }


    /*
     *  Insertion Sort List
     *  Sort a linked list using insertion sort.
     */

    public ListNode insertionSortList(ListNode head) {
        if (head==null || head.next==null) {
            return head;
        }

        ListNode preHead = new ListNode (-1);

        preHead.next = head;
        ListNode run = head;


        while (run != null && run.next != null) {
            if(run.val > run.next.val) {
                ListNode smallNode = run.next;
                ListNode pre = preHead;

                while(pre.next.val < smallNode.val) {
                    pre = pre.next;
                }

                run.next = smallNode.next;
                smallNode.next = pre.next;
                pre.next = smallNode;

            } else {
                run = run.next;
            }

        }

        return preHead.next;
    }


    /*
     *  Intersection of Two Linked Lists
     *  Write a program to find the node at which the intersection of two singly linked lists begins.
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int lengthA = 0;
        int lengthB = 0;
        ListNode nodeA = headA;
        ListNode nodeB = headB;
        while (nodeA != null) {
            lengthA++;
            nodeA = nodeA.next;

        }

        while (nodeB != null) {
            lengthB++;
            nodeB = nodeB.next;

        }

        nodeA = headA;
        nodeB = headB;

        if (lengthA > lengthB) {
            int i=0;
            while(i<lengthA-lengthB){
                nodeA = nodeA.next;
                i++;
            }
        } else {
            int i=0;
            while(i<lengthB-lengthA) {
                nodeB = nodeB.next;
                i++;
            }
        }

        while (nodeA != null && nodeB != null) {
            if(nodeA.val == nodeB.val) {
                return nodeB;
            }
            nodeA = nodeA.next;
            nodeB = nodeB.next;
        }

        return null;
    }


    /*
     *
     *   Implement strStr()
     *   The signature of the function had been updated to return the index instead of the pointer.
     *   If you still see your function signature returns a char
     *
     */
    public int strStr(String haystack, String needle) {
        if (haystack.length() < needle.length()) {
            return -1;
        }
        for (int i=0; i<=haystack.length() - needle.length(); i++) {
            if (haystack.substring(i, i+needle.length()).equals(needle)) {
                return i;
            }
        }
        return -1;
    }



    /*
     *
     *   Path Sum II
     *   Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.
     *
     */

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        if (root == null) {
            return result;
        }
        dfs(result, root, new LinkedList<Integer>(), sum);
        return result;

    }

    public void dfs (List<List<Integer>> result, TreeNode node, List<Integer> temp, int sum) {
        if (node.left == null && node.right == null) {
            if (node.val == sum) {
                temp.add(node.val);
                result.add(temp);
            }
        } else {
            if (node.left != null) {
                List<Integer> list = new LinkedList<Integer>(temp);
                list.add(node.val);
                dfs(result, node.left, list, sum-node.val);
            }

            if (node.right != null) {
                List<Integer> list = new LinkedList<Integer>(temp);
                list.add(node.val);
                dfs(result, node.right, list, sum-node.val);
            }
        }
    }


    /*
     *
     *   Find Peak Element
     *   Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.
     *   For example, in array [1, 2, 3, 1], 3 is a peak element and your function should return the index number 2.
     */

    public int findPeakElement(int[] nums) {
        for(int i = 1; i < nums.length; i ++) {
            if(nums[i] < nums[i-1]) {
                return i-1;
            }
        }
        return nums.length-1;
    }


    /*
     *
     *   Invert Binary Tree
     */

    public TreeNode invertTree(TreeNode root) {
        if (root != null) {
            invert(root);
        }

        return root;
    }

    public void invert(TreeNode root) {
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        if(root.left != null) {
            invert(root.left);
        }

        if(root.right != null) {
            invert(root.right);
        }
    }


    /*
     *   Isomorphic Strings
     *   Given "egg", "add", return true. Given "paper", "title", return true.  Given "foo", "bar", return false.
     */
    public boolean isIsomorphic(String s, String t) {
        if(s.length() != t.length()) {
            return false;
        } else {
            Map map1 = new HashMap<Character, Character>();
            Map map2 = new HashMap<Character, Character>();

            for(int i=0; i<s.length(); i++) {
                Character c1 = s.charAt(i);
                Character c2 = t.charAt(i);
                if(map1.containsKey(c1)) {
                    if(map1.get(c1) != c2) {
                        return false;
                    }
                } else if(map2.containsKey(c2)) {
                    if(map2.get(c2) != c1) {
                        return false;
                    }
                }

                map1.put(c1, c2);
                map2.put(c2, c1);
            }

            return true;
        }
    }


    /*
     *   Kth Smallest Element in a BST
     *   Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
     */

    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<TreeNode>();

        TreeNode p = root;
        int result = 0;

        while (!stack.isEmpty() || p != null) {
            if(p != null) {
                stack.push(p);
                p = p.left;
            } else {
                TreeNode t = stack.pop();
                k--;
                if(k == 0) {
                    result = t.val;
                }
                p = t.right;
            }
        }

        return result;
    }


    /*
     *   Length of Last Word
     *   Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.
     *   Given s = "Hello World", return 5.
     */

    public int lengthOfLastWord(String s) {
        if(s == null || s.length() == 0) {
            return 0;
        }

        int index1 = s.length() - 1;
        while(index1 >= 0 && s.charAt(index1) == ' ') {
            index1--;
        }
        int index2 = index1;
        while(index2 >= 0 && s.charAt(index2) != ' ') {
            index2--;
        }

        return index1 - index2;
    }


    /*
     *   Binary Tree Level Order Traversal
     *   Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).
     */

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        List<Integer> values = new LinkedList<Integer>();

        LinkedList<TreeNode> current = new LinkedList<TreeNode>();
        LinkedList<TreeNode> next = new LinkedList<TreeNode>();

        if(root == null) {
            return result;
        }

        current.add(root);

        while(!current.isEmpty()) {
            TreeNode temp = current.remove();
            values.add(temp.val);
            if(temp.left != null) {
                next.add(temp.left);
            }

            if(temp.right != null) {
                next.add(temp.right);
            }

            if(current.isEmpty()) {
                current = next;
                result.add(values);
                values = new LinkedList<Integer>();
                next = new LinkedList<TreeNode>();
            }
        }
        return result;
    }


    /*
     *  Binary Tree Level Order Traversal II
     *  Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).
     *  Given binary tree {3,9,20,#,#,15,7},
     */

    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        LinkedList<List<Integer>> list = new LinkedList<List<Integer>>();
        LinkedList<Integer> values = new LinkedList<Integer>();

        if(root == null) {
            return list;
        }

        LinkedList<TreeNode> current = new LinkedList<TreeNode>();
        LinkedList<TreeNode> next = new LinkedList<TreeNode>();
        current.add(root);
        while(!current.isEmpty()) {
            TreeNode tmp = current.remove();
            values.add(tmp.val);
            if(tmp.left != null) {
                next.add(tmp.left);
            }

            if(tmp.right != null) {
                next.add(tmp.right);
            }

            if(current.isEmpty()) {
                current = next;
                list.add(values);
                next = new LinkedList<TreeNode>();
                values = new LinkedList<Integer>();
            }
        }

        LinkedList<List<Integer>> result = new LinkedList<List<Integer>>();

        while(!list.isEmpty()) {
            result.add(list.pollLast());
        }

        return result;
    }



    /*
     *   Recover Binary Search Tree
     *   Two elements of a binary search tree (BST) are swapped by mistake.
     *   Recover the tree without changing its structure.
     */


    public void recoverTree(TreeNode root) {
        TreeNode current = root;
        TreeNode prev = null;
        TreeNode node1 = null;
        TreeNode node2 = null;
        while (current != null) {
            if (current.left == null) {
                if (prev != null) {
                    if (prev.val >= current.val) {
                        if (node1 == null)
                            node1 = prev;
                        node2 = current;
                    }
                }
                prev = current;
                current = current.right;
            } else {
                TreeNode t = current.left;
                while (t.right != null && t.right != current)
                    t = t.right;
                if (t.right == null) {
                    t.right = current;
                    current = current.left;
                } else {
                    t.right = null;
                    if (prev != null) {
                        if (prev.val >= current.val) {
                            if (node1 == null)
                                node1 = prev;
                            node2 = current;
                        }
                    }
                    prev = current;
                    current = current.right;
                }
            }
        }
        int tmp = node1.val;
        node1.val = node2.val;
        node2.val = tmp;
    }


    /*
     *   Lowest Common Ancestor of a Binary Search Tree
     *   Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
     *
     */

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestor(root.right, p, q);
        } else if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestor(root.left, p, q);
        } else {
            return root;
        }
    }

    /*
     *  Lowest Common Ancestor of a Binary Tree
     *  Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
     */
     public TreeNode lowestCommonAncestor1(TreeNode root, TreeNode p, TreeNode q) {
         if (root == null) {
             return null;
         }
         if (root == p || root == q) {
             return root;
         }

         TreeNode left = lowestCommonAncestor1(root.left, p, q);
         TreeNode right = lowestCommonAncestor1(root.right, p, q);

         if (left != null && right != null) {
             return root;
         } else if (left != null) {
             return left;
         } else if (right != null) {
             return right;
         } else {
             return null;
         }
     }

    /*
     *  Majority Element
     *  Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
     *  You may assume that the array is non-empty and the majority element always exist in the array.
     */
    public int majorityElement(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }

        Arrays.sort(nums);
        return nums[nums.length / 2];
    }

    /*
     *   Minimum Depth of Binary Tree
     *   Given a binary tree, find its minimum depth.
     *
     */
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        } else if (root.left == null) {
            return minDepth(root.right) + 1;
        } else if (root.right == null) {
            return minDepth(root.left) + 1;
        } else {
            return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
        }
    }

    /*
     *   Maximum Depth of Binary Tree
     *   Given a binary tree, find its maximum depth.
     *   The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
     */
    public int maxDepth(TreeNode root) {
        if(root == null) {
            return 0;
        } else {
            return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
        }
    }

    /*
     *   Delete Node in a Linked List
     *   Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are gi
     *   ven the third node with value 3, the linked list should become 1 -> 2 -> 4
     *
     */
    public void deleteNode(ListNode node) {
        ListNode temp = node.next;
        node.val = temp.val;
        node.next = temp.next;
    }



    /*
     *   Binary Tree Paths
     *   Given a binary tree, return all root-to-leaf paths.
     *
     */

    public List<String> binaryTreePaths(TreeNode root) {
        List<String> result = new LinkedList<String>();
        if (root == null) {
            return result;
        }

        dfs(root, new StringBuilder(), result);

        return result;
    }


    public void dfs(TreeNode node, StringBuilder sb, List<String> result) {
        if(node.left == null && node.right == null) {
            sb.append(node.val);
            result.add(sb.toString());
            return;
        }

        sb.append(node.val);
        sb.append("->");

        if(node.left != null) {
            dfs(node.left, new StringBuilder(sb), result);
        }

        if(node.right != null) {
            dfs(node.right, new StringBuilder(sb), result);
        }
    }


    /*
     *   Add Binary
     *   Given two binary strings, return their sum (also a binary string).
     *   a = "11", b="1" return "100"
     */

    public String addBinary(String a, String b) {
        int lenA = a.length(), lenB = b.length();
        int i = lenA - 1, j = lenB - 1;
        StringBuilder sb = new StringBuilder();
        int carry = 0;
        while (i >= 0 || j >= 0) {
            int va = i >= 0? Integer.parseInt(String.valueOf(a.charAt(i))) : 0;
            int vb = j >= 0? Integer.parseInt(String.valueOf(b.charAt(j))) : 0;
            int sum = va + vb + carry;
            sb.insert(0, sum%2);
            carry = sum/2;
            i--;
            j--;
        }

        if (carry > 0) {
            sb.insert(0, '1');
        }

        return sb.toString();
    }


    /*
     *   Binary Tree Inorder Traversal
     *   Given a binary tree, return the inorder traversal of its nodes' values.
     *
     */

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new LinkedList<Integer>();
        if(root == null) {
            return result;
        } else {
            inorderTraversal(root, result);
            return result;
        }
    }

    public void inorderTraversal(TreeNode node, List<Integer> result) {
        if (node.left != null) {
            inorderTraversal(node.left, result);
        }
        result.add(node.val);
        if (node.right != null) {
            inorderTraversal(node.right, result);
        }
    }


    /*
     *   Merge Sorted Array
     *   Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
     *
     */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        while(m > 0 && n > 0) {
            if(nums1[m-1] < nums2[n-1]) {
                nums1[m+n-1] = nums2[n-1];
                n--;
            } else {
                nums1[m+n-1] = nums1[m-1];
                m--;
            }
        }

        while(n > 0) {
            nums1[m+n-1] = nums2[n-1];
            n--;
        }
    }


    /*
     *   Min Stack
     *   Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
     *
     */
    class MinStack {
        long min;
        Stack<Long> stack;

        public MinStack() {
            stack = new Stack<Long>();
        }

        public void push(int x) {
            if (stack.isEmpty()) {
                stack.push(0L);
                min = x;
            } else {
                stack.push(x - min);//Could be negative if min value needs to change
                if (x < min) {
                    min = x;
                }
            }
        }

        public void pop() {
            if (stack.isEmpty()) {
                return;
            }
            long pop = stack.pop();
            if (pop < 0) {
                min = min - pop;//If negative, increase the min value
            }
        }

        public int top() {
            long top = stack.peek();
            if (top > 0) {
                return (int)(top + min);
            } else {
                return (int)(min);
            }
        }

        public int getMin() {
            return (int)min;
        }
    }


    /*
     *  Minimum Size Subarray Sum
     *  For example, given the array [2,3,1,2,4,3] and s = 7,
     *  the subarray [4,3] has the minimal length under the problem constraint.
     */
    public int minSubArrayLen(int s, int[] nums) {
        int start = 0;
        int end = 0;
        int minLen = nums.length;
        int sum = 0;
        while (end < nums.length) {
            sum += nums[end];
            while (sum >= s) {
                if (start == end) {
                    return 1;
                } else {
                    minLen = Math.min(minLen, end-start+1);
                    sum -= nums[start++];
                }

            }
            end++;
        }
        return minLen == nums.length? 0: minLen;
    }


    /*
     *   Merge Two Sorted Lists
     *
     *
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode helper = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                helper.next = l1;

                l1 = l1.next;
            } else {
                helper.next = l2;
                l2 = l2.next;
            }
            helper = helper.next;
        }
        if (l1 != null) {
            helper.next = l1;
        }
        if (l2 != null) {
            helper.next = l2;
        }
        return dummy.next;

    }



    /*
     *  Balanced Binary Tree
     *  Given a binary tree, determine if it is height-balanced.
     *  For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by
     */

    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        } else if (root.left == null && root.right == null) {
            return true;
        } else if (Math.abs(depth(root.left) - depth(root.right)) > 1) {
            return false;
        } else {
            return isBalanced(root.left) && isBalanced(root.right);
        }

    }

    public int depth(TreeNode node) {
        if(node == null) {
            return 0;
        }

        return 1 + Math.max(depth(node.left), depth(node.right));
    }



    /*
     *  Binary Search Tree Iterator
     *  Calling next() will return the next smallest number in the BST.
     *  Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
     */
    public class BSTIterator {
        Stack<TreeNode> stack;
        public BSTIterator(TreeNode root) {
            stack = new Stack<TreeNode>();
            while (root != null) {
                stack.push(root);
                root = root.left;
            }
        }

        /** @return whether we have a next smallest number */
        public boolean hasNext() {
            return !stack.isEmpty();
        }

        /** @return the next smallest number */
        public int next() {
            TreeNode node = stack.pop();
            int result = node.val;
            if (node.right != null) {
                node = node.right;
                while (node != null) {
                    stack.push(node);
                    node = node.left;
                }
            }
            return result;
        }
    }


    /*
     *   Rotate List
     *   Given a list, rotate the list to the right by k places, where k is non-negative.
     *   Given 1->2->3->4->5->NULL and k = 2,
     *   return 4->5->1->2->3->NULL.
     *
     */

    public ListNode rotateRight(ListNode head, int k) {

        if (head == null || k == 0) {
            return head;
        }

        int len = 1;
        ListNode runner = head;
        while(runner.next != null) {
            runner = runner.next;
            len++;
        }

        int n = len - k%len;
        runner.next = head;

        while(n > 0) {
            runner = runner.next;
            n--;
        }

        head = runner.next;
        runner.next = null;

        return head;
    }


    /*
     *  Swap Nodes in Pairs
     *  Given a linked list, swap every two adjacent nodes and return its head.
     *  Given 1->2->3->4, you should return the list as 2->1->4->3.
     */
    public ListNode swapPairs(ListNode head) {
        ListNode curr = head;
        while (curr != null && curr.next != null) {
            ListNode next = curr.next;
            int temp = curr.val;
            curr.val = next.val;
            next.val = temp;
            if (next != null) {
                curr = next.next;
            }
        }
        return head;
    }


     /*
     *  Reverse Linked List
     *  Reverse a singly linked list.
     *
     */
     public ListNode reverseList(ListNode head) {
         ListNode helper = null;
         ListNode current = head;
         while(current != null) {
             ListNode next = current.next;
             current.next = helper;
             helper = current;
             current = next;
         }

         return helper;
     }


    /*
     *  Climbing Stairs
     *  Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
     *
     */
    public int climbStairs(int n) {
        if (n == 1 || n == 2) {
            return n;
        }

        int[] arr = new int[n];
        arr[0] = 1;
        arr[1] = 2;
        for (int i=2; i<n; i++) {
            arr[i] = arr[i-1] + arr[i-2];
        }
        return arr[n-1];

    }


     /*
     *  Reverse Nodes in k-Group
     *  Given this linked list: 1->2->3->4->5
     *  For k = 2, you should return: 2->1->4->3->5
     *  For k = 3, you should return: 3->2->1->4->5
     */

    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode preHead = new ListNode(-1);
        preHead.next = head;
        ListNode runner = preHead;
        ListNode curr = preHead;
        int count = 0;
        Stack<Integer> s = new Stack<Integer>();
        while (runner.next != null) {
            runner = runner.next;
            count++;
            s.push(runner.val);
            if (count == k) {
                while (!s.isEmpty()) {
                    curr = curr.next;
                    curr.val = s.pop();
                }
                count = 0;
            }
        }
        return head;
    }


     /*
     *  Reverse Linked List II
     *  Reverse a linked list from position m to n. Do it in-place and in one-pass.
     *  Given 1->2->3->4->5->NULL, m = 2 and n = 4,
     *  For k = 3, you should return: 3->2->1->4->5
     *  return 1->4->3->2->5->NULL.
     */

    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode pre = new ListNode(-1);
        pre.next = head;
        head = pre;
        int count = m-1;
        while(pre.next != null && count > 0) {
            pre = pre.next;
            count--;
        }

        ListNode curr = pre.next;
        int k = n-m;
        while(curr.next != null && k>0) {
            ListNode temp = curr.next;
            curr.next = temp.next;
            temp.next = pre.next;
            pre.next = temp;
            k--;
        }

        return head.next;
    }

    public ListNode reverseBetween2ndVersion(ListNode head, int m, int n) {
        ListNode pre = new ListNode(-1);
        pre.next = head;
        ListNode runner = pre;
        Stack<Integer> s = new Stack<Integer>();
        int count = 0;
        while (runner != null) {
            runner = runner.next;
            count++;
            if (count >= m && count <= n) {
                s.push(runner.val);
            }
        }
        runner = pre;
        count = 0;
        while (runner != null) {
            runner = runner.next;
            count++;
            if (count >= m && count <= n) {
                runner.val = s.pop();
            }
        }

        return head;
    }


    /*
     *  Remove Nth Node From End of List
     *  Given a linked list, remove the nth node from the end of list and return its head.
     *  Given linked list: 1->2->3->4->5, and n = 2.
     *  After removing the second node from the end, the linked list becomes 1->2->3->5.
     *
     */

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode n1 = dummy;
        ListNode n2 = dummy;
        int count = 0;
        while( n1.next != null) {
            if(count >= n) {
                n2 = n2.next;
            }
            n1 = n1.next;
            count++;
        }

        if(n2.next != null) {
            n2.next = n2.next.next;
        }

        return dummy.next;

    }


    /*
     *  Valid Parentheses
     *  Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
     *  The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
     *
     */

    public boolean isValid(String s) {
        HashMap<Character, Character> map = new HashMap<Character, Character>();
        map.put('(', ')');
        map.put('[', ']');
        map.put('{', '}');
        Stack<Character> stack = new Stack<Character>();
        for(int i=0; i<s.length(); i++) {
            char curr = s.charAt(i);
            if(map.keySet().contains(curr)) {
                stack.push(curr);
            } else if (map.values().contains(curr)) {
                if(!stack.isEmpty() && map.get(stack.peek()) == curr) {
                    stack.pop();
                } else {
                    return false;
                }
            }
        }

        return stack.empty();
    }


    /*
     *  Remove Linked List Elements
     *  Given: 1 --> 2 --> 6 --> 3 --> 4 --> 5 --> 6, val = 6
     *  Return: 1 --> 2 --> 3 --> 4 --> 5
     *
     */

    public ListNode removeElements(ListNode head, int val) {
        ListNode helper = new ListNode(0);
        helper.next = head;
        ListNode current = helper;
        while(current.next != null) {
            if(current.next.val == val) {
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }

        return helper.next;
    }


    /*
     *  Remove Duplicates from Sorted List
     *  Given 1->1->2, return 1->2.
     *  Given 1->1->2->3->3, return 1->2->3.
     *
     */
    public ListNode deleteDuplicatesI(ListNode head) {
        ListNode current = head;
        while(current != null && current.next != null) {
            if(current.val == current.next.val) {
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }

        return head;
    }



    /*
     *  Remove Duplicates from Sorted List II
     *  Given 1->2->3->3->4->4->5, return 1->2->5
     *  Given 1->1->1->2->3, return 2->3
     *
     */

    public ListNode deleteDuplicates(ListNode head) {
        ListNode helper = new ListNode(-1);
        helper.next = head;
        head = helper;
        while (helper.next != null && helper.next.next != null) {
            if (helper.next.val == helper.next.next.val) {
                int dup = helper.next.val;
                while (helper.next != null && helper.next.val == dup) {
                    helper.next = helper.next.next;
                }
            } else {
                helper = helper.next;
            }
        }

        return head.next;
    }


    /*
     *  Triangle
     *  Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below
     *
     *
     */

    public int minimumTotal(List<List<Integer>> triangle) {
        int[] total = new int[triangle.size()];
        int l = triangle.size() - 1;

        for (int i = 0; i < triangle.get(l).size(); i++) {
            total[i] = triangle.get(l).get(i);
        }

        // iterate from last second row
        for (int i = triangle.size() - 2; i >= 0; i--) {
            for (int j = 0; j < triangle.get(i + 1).size() - 1; j++) {
                total[j] = triangle.get(i).get(j) + Math.min(total[j], total[j + 1]);
            }
        }

        return total[0];
    }



    /*
     *  Permutations
     *  Given a collection of numbers, return all possible permutations.
     *  [1,2,3] have the following permutations:
     *  [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], and [3,2,1].
     */

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        result.add(new LinkedList<Integer>());
        for (int i=0; i<nums.length; i++) {
            List<List<Integer>> current = new LinkedList<List<Integer>>();
            for (List<Integer> list: result) {
                for (int j=0; j<list.size()+1; j++) {
                    list.add(j, nums[i]);
                    current.add(new LinkedList<Integer>(list));
                    list.remove(j);
                }
            }
            result = current;
        }
        return result;
    }


    /*
     *  Permutations II
     *  Given a collection of numbers that might contain duplicates, return all possible unique permutations.
     *  [1,1,2] have the following unique permutations:
     *  [1,1,2], [1,2,1], and [2,1,1].
     */

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        result.add(new LinkedList());
        for (int i=0; i<nums.length; i++) {
            Set<List<Integer>> curr = new HashSet<List<Integer>>();
            for (List<Integer> list : result) {
                for (int j=0; j<list.size()+1; j++) {
                    list.add(j, nums[i]);
                    curr.add(new LinkedList<Integer>(list));
                    list.remove(j);
                }
            }
            result = new LinkedList<List<Integer>>(curr);
        }

        return result;
    }



    /*
     *  Add Digits
     *  Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
     *  Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.
     */
    public int addDigits(int num) {
        return (num - 1) % 9 + 1;
    }


    /*
     *  Single Number III
     *  Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the
     *  two elements that appear only once.
     *  Given nums = [1, 2, 1, 3, 2, 5], return [3, 5].
     */

    public int[] singleNumber3(int[] nums) {
        int A = 0;
        int B = 0;
        int AXORB = 0;
        for(int i = 0; i<nums.length; i++){
            AXORB ^= nums[i];
        }

        AXORB = (AXORB & (AXORB - 1)) ^ AXORB; //find the different bit
        for(int i = 0; i<nums.length; i++){
            if((AXORB & nums[i]) == 0)
                A ^= nums[i];
            else
                B ^= nums[i];
        }
        return new int[]{A, B};

    }


    /*
     *  Combination Sum
     *  For example, given candidate set 2,3,6,7 and target 7,
     *  A solution set is: [7] [2,2,3]
     *
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        if (candidates == null || candidates.length == 0) {
            return result;
        }

        Arrays.sort(candidates);

        List<Integer> current = new LinkedList<Integer>();
        combinationSum(candidates, result, current, target, 0);
        return result;
    }

    public void combinationSum(int[] candidates, List<List<Integer>> result,
                               List<Integer> current, int target, int index) {

        if (target == 0) {
            result.add(current);
        } else {
            for (int i=index; i<candidates.length; i++) {
                if (target < candidates[i]) {
                    break;
                }
                List<Integer> temp = new LinkedList<Integer>(current);
                temp.add(candidates[i]);
                combinationSum(candidates, result, temp, target - candidates[i], i);
            }
        }

    }


    /*
     *  Combination Sum III
     *  Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each
     *  Input: k = 3, n = 7
     *  [[1,2,4]]
     */

    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        rec(result, new LinkedList<Integer>(), k, n, 1);
        return result;
    }

    public void rec(List<List<Integer>> result, List<Integer> temp, int k, int n, int pos) {
        if (n == 0 && temp.size() == k) {
            result.add(temp);
        } else {
            for (int i=pos; i<=9; i++) {
                if (n-i >= 0) {
                    List<Integer> list = new LinkedList<Integer>(temp);
                    list.add(i);
                    rec(result, list, k, n-i, i+1);
                }
            }
        }
    }


    /*
     *  Maximum Subarray
     *  Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
     *  For example, given the array [−2,1,−3,4,−1,2,1,−5,4],
     *  the contiguous subarray [4,−1,2,1] has the largest sum = 6.
     */

    public int maxSubArray(int[] nums) {
        int newSum = nums[0];
        int max = nums[0];
        for (int i=1; i<nums.length; i++) {
            newSum = Math.max(newSum+nums[i], nums[i]);
            max = Math.max(max, newSum);
        }

        return max;
    }


    /*
     *  Multiply Strings
     *  Given two numbers represented as strings, return multiplication of the numbers as a string.
     *  Note: The numbers can be arbitrarily large and are non-negative.
     *
     */

    public String multiply(String num1, String num2) {
        String n1 = new StringBuilder(num1).reverse().toString();
        String n2 = new StringBuilder(num2).reverse().toString();
        int[] arr = new int[num1.length() + num2.length()];
        for (int i=0; i<n1.length(); i++) {
            for (int j=0; j<n2.length(); j++) {
                arr[i+j] += (n1.charAt(i) - '0') * (n2.charAt(j) - '0');
            }
        }
        StringBuilder sb = new StringBuilder();
        int carry = 0;
        for (int k=0; k<arr.length; k++) {
            if (arr[k] >= 10 && k < arr.length-1) {
                arr[k+1] += arr[k]/10;
                arr[k] %= 10;
            }
            sb.insert(0, arr[k] + "");
        }

        while (sb.charAt(0) == '0' && sb.length() > 1) {
            sb.deleteCharAt(0);
        }
        return sb.toString();
    }


    /*
     *  String to Integer (atoi)
     *  Implement atoi to convert a string to an integer.
     *
     *
     */

    public int myAtoi(String str) {
        if (str == null || str.length() == 0) {
            return 0;
        }


        // trim white spaces
        str = str.trim();

        char flag = '+';

        // check negative or positive
        int i = 0;

        if (str.charAt(0) == '-') {
            flag = '-';
            i++;
        } else if (str.charAt(0) == '+') {
            i++;
        }
        // use double to store result
        double result = 0;

        // calculate value
        while (str.length() > i && str.charAt(i) >= '0' && str.charAt(i) <= '9') {
            result = result * 10 + (str.charAt(i) - '0');
            i++;
        }

        if (flag == '-')
            result = -result;

        return (int) result;
    }


    /*
     *  Palindrome Number
     *  Determine whether an integer is a palindrome. Do this without extra space.
     */
    public boolean isPalindrome(int x) {
        String num = x + "";
        int i = 0;
        int j= num.length() - 1;
        while (i <= j) {
            if (num.charAt(i) != num.charAt(j)) {
                return false;
            }

            i++;
            j--;
        }

        return true;
    }

    /*
     *  Palindrome Linked List
     *  Given a singly linked list, determine if it is a palindrome.
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null) {
            return true;
        }
        ListNode helper = head;
        ListNode newHead = null;
        while (helper != null) {
            ListNode temp = new ListNode(helper.val);
            temp.next = newHead;
            newHead = temp;
            helper = helper.next;
        }
        ListNode p1 = head;
        ListNode p2 = newHead;
        while (p1 != null) {
            if (p1.val != p2.val) {
                return false;
            }
            p1 = p1.next;
            p2 = p2.next;
        }

        return true;
    }


    /*
     *  Longest Common Prefix
     *  Write a function to find the longest common prefix string amongst an array of strings.
     */

    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }

        int minLength = strs[0].length();
        for (int i=1; i<strs.length; i++) {
            if (strs[i].length() < minLength) {
                minLength = strs[i].length();
            }
        }

        for (int i=0; i<minLength; i++) {
            char temp = strs[0].charAt(i);
            for(int j=0; j<strs.length; j++) {
                if(strs[j].charAt(i) != temp) {
                    return strs[0].substring(0, i);
                }
            }
        }
        return strs[0].substring(0, minLength);
    }




    /*
     *  Power of Two
     *  Given an integer, write a function to determine if it is a power of two.
     */
    public boolean isPowerOfTwo(int n) {
        return n > 0 && ((n & (n - 1)) == 0 );
    }



    /*
     *  Plus One
     *  Given a non-negative number represented as an array of digits, plus one to the number.
     */

    public int[] plusOne(int[] digits) {
        for (int i=digits.length-1; i>=0; i--) {
            if (digits[i] + 1 > 9) {
                digits[i] = 0;
            } else {
                digits[i] += 1;
                break;
            }
        }

        if (digits[0] == 0) {
            int[] result = new int[digits.length+1];
            result[0] = 1;
            return result;
        }

        return digits;
    }

    /*
     *  Remove Duplicates from Sorted Array
     *  Given input array nums = [1,1,2],  Your function should return length = 2
     */

    public int removeDuplicates(int[] nums) {
        int i = 0;
        int j = 1;
        while(j < nums.length) {
            if(nums[i] != nums[j]) {
                i++;
                nums[i] = nums[j];
            }
            j++;
        }
        return i+1;
    }

    /*
     *  Remove Element
     *  Given an array and a value, remove all instances of that value in place and return the new length.
     */
    public int removeElement(int[] nums, int val) {
        int i=0, j=0;
        while (j < nums.length) {
            if (nums[j] != val) {
                nums[i] = nums[j];
                i++;
            }

            j++;
        }

        return i;
    }


    /*
     *
     *   Summary Ranges
     *   For example, given [0,1,2,4,5,7], return ["0->2","4->5","7"].
     *
     */
    public List<String> summaryRanges(int[] nums) {
        List<String> list = new LinkedList<String>();
        if(nums==null || nums.length == 0) {
            return  list;
        }
        int start = 0;
        int end = 0;
        while(end < nums.length) {
            if(end < nums.length - 1 && nums[end+1] - nums[end] == 1) {
                end++;
            } else {
                if(start == end) {
                    list.add("" + nums[end]);
                } else {
                    list.add(nums[start] + "->" + nums[end]);
                }
                end++;
                start = end;
            }
        }
        return list;
    }

    /*
     *  Valid Palindrome
     *  "A man, a plan, a canal: Panama" is a palindrome.  "race a car" is not a palindrome.
     */
    public boolean isPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            while(i < j && !Character.isLetterOrDigit(s.charAt(i))) {
                i++;
            }
            while (i < j && !Character.isLetterOrDigit(s.charAt(j))) {
                j--;
            }
            if (Character.toLowerCase(s.charAt(i)) != Character.toLowerCase(s.charAt(j))) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }

    /*
     *  Word Pattern
     *  pattern = "abba", str = "dog cat cat dog" should return true.
     *  pattern = "abba", str = "dog cat cat fish" should return false.
     */
    public boolean wordPattern(String pattern, String str) {
        char[] keys = pattern.toCharArray();
        String[] values = str.split(" ");
        if (keys.length != values.length) {
            return false;
        }
        Map<Character, String> map = new HashMap<Character, String>();
        Set<String> set = new HashSet<String>();
        for (int i=0; i<keys.length; i++) {
            if (!map.containsKey(keys[i]) && !set.contains(values[i])) {
                set.add(values[i]);
                map.put(keys[i], values[i]);
            } else if (map.containsKey(keys[i]) && map.get(keys[i]).equals(values[i])) {
                continue;
            } else {
                return false;
            }
        }
        return true;
    }


    /*
     *  Search for a Range
     *  Given [5, 7, 7, 8, 8, 10] and target value 8,
     *  return [3, 4].
     */
    public int[] searchRange(int[] nums, int target) {
        int count = 0, start = -1;
        for (int i=0; i<nums.length; i++) {
            if (nums[i] == target) {
                if (start == -1) {
                    start = i;
                }
                count ++;
            }
        }
        if (count > 0) {
            return new int[]{start, start+count-1};
        } else {
            return new int[]{-1, -1};
        }
    }

    /*
     *  Word Break
     *  s = "leetcode", dict = ["leet", "code"].
     *  Return true because "leetcode" can be segmented as "leet code".
     */
    public boolean wordBreak(String s, Set<String> wordDict) {
        int[] flags = new int[s.length()+1];
        flags[0] = 1;
        for (int i=0; i<s.length(); i++) {
            if (flags[i] == 1) {
                for (String str : wordDict) {
                    int end = i + str.length();
                    if (end <= s.length()) {
                        if (s.substring(i, end).equals(str)) {
                            flags[end] = 1;
                        }
                    }
                }
            }
        }

        return flags[s.length()] == 1;
    }


    /*
     *
     *   Reverse Integer
     *   Example1: x = 123, return 321
     *   Example2: x = -123, return -321
     */
    public int reverse(int x) {
        int a = x;
        if (x < 0) {
            a = 0 - a;
        }

        int temp = a;
        int count = 0;
        while (temp > 0) {
            temp /= 10;
            count++;
        }
        temp = a;
        long sum = 0;
        while(temp > 0) {
            int mod = temp % 10;
            sum += mod * Math.pow(10, count-1);
            temp /= 10;
            count--;
        }

        if (sum > 2147483647) {
            return 0;
        }

        if (x < 0) {
            sum = 0 - sum;
        }

        return (int) sum;

    }

    /*
     *  Missing Number
     *  Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.
     *  Given nums = [0, 1, 3] return 2.
     */
    public int missingNumber(int[] nums) {
        int len = nums.length;
        int sum = len*(len+1)/2;
        for (int i=0; i<nums.length; i++) {
            sum -= nums[i];
        }
        return sum;
    }

    /*
     *  First Bad Version
     *  Suppose you have n versions [1, 2, ..., n] and you want
     *  to find out the first bad one, which causes all the following ones to be bad.
     */
    public int firstBadVersion(int n) {
        int start = 1, end = n, mid;
        while (start + 1 < end) {
            mid = start + (end - start)/2;
            if (isBadVersion(mid)) {
                end = mid;
            } else {
                start = mid;
            }
        }

        if (isBadVersion(start)) {
            return start;
        } else if (isBadVersion(end)) {
            return end;
        } else {
            return -1; // not found
        }
    }

    boolean isBadVersion(int version) {
        return false;
    }


    /*
     *  Linked List Cycle
     *  Given a linked list, determine if it has a cycle in it.
     */
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;

        while(fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;

            if(slow == fast) {
                return true;
            }
        }
        return false;
    }

    /*
     *  Binary Tree Preorder Traversal
     *  Given a binary tree, return the preorder traversal of its nodes' values.
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new LinkedList<Integer>();
        if (root == null) {
            return result;
        }
        preOrder(root, result);
        return result;
    }

    public void preOrder(TreeNode node, List<Integer> result) {
        if (node != null) {
            result.add(node.val);
        }
        if (node.left != null) {
            preOrder(node.left, result);
        }

        if (node.right != null) {
            preOrder(node.right, result);
        }
    }

    /*
     *  Best Time to Buy and Sell Stock
     *  f you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.
     */
    public int maxProfit(int[] prices) {
        int maxPro = 0;
        int minPrice = Integer.MAX_VALUE;
        for(int i = 0; i < prices.length; i++){
            minPrice = Math.min(minPrice, prices[i]);
            maxPro = Math.max(maxPro, prices[i] - minPrice);
        }
        return maxPro;
    }

    /*
     *  Best Time to Buy and Sell Stock II
     *  Say you have an array for which the ith element is the price of a given stock on day i.
     */
    public int maxProfit2(int[] prices) {
        int profit = 0;
        for(int i=1; i<prices.length; i++) {
            int diff = prices[i]-prices[i-1];
            if(diff > 0) {
                profit += diff;
            }
        }
        return profit;
    }

    /*
     *  Decode Ways
     *  A message containing letters from A-Z is being encoded to numbers using the following mapping:
     */
    public int numDecodings(String s) {
        if(s == null || s.length() == 0 || s.equals("0")) {
            return 0;
        }
        int[] t = new int[s.length()+1];
        t[0] = 1;
        if(isValidString(s.substring(0,1))) {
            t[1] = 1;
        } else {
            t[1] = 0;
        }

        for(int i=2; i<=s.length(); i++) {
            if(isValidString(s.substring(i-1,i))) {
                t[i] += t[i-1];
            }

            if(isValidString(s.substring(i-2,i))) {
                t[i] += t[i-2];
            }
        }

        return t[s.length()];
    }

    public boolean isValidString(String s) {
        if (s.charAt(0) == '0') {
            return false;
        }
        int value = Integer.parseInt(s);
        return value >= 1 && value <= 26;
    }


    /*
     *  Basic Calculator
     *  Implement a basic calculator to evaluate a simple expression string.
     */
    public int calculate(String s) {
        // delte white spaces
        s = s.replaceAll(" ", "");

        Stack<String> stack = new Stack<String>();
        char[] arr = s.toCharArray();

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == ' ')
                continue;

            if (arr[i] >= '0' && arr[i] <= '9') {
                sb.append(arr[i]);

                if (i == arr.length - 1) {
                    stack.push(sb.toString());
                }
            } else {
                if (sb.length() > 0) {
                    stack.push(sb.toString());
                    sb = new StringBuilder();
                }

                if (arr[i] != ')') {
                    stack.push(new String(new char[] { arr[i] }));
                } else {
                    // when meet ')', pop and calculate
                    ArrayList<String> t = new ArrayList<String>();
                    while (!stack.isEmpty()) {
                        String top = stack.pop();
                        if (top.equals("(")) {
                            break;
                        } else {
                            t.add(0, top);
                        }
                    }

                    int temp = 0;
                    if (t.size() == 1) {
                        temp = Integer.valueOf(t.get(0));
                    } else {
                        for (int j = t.size() - 1; j > 0; j = j - 2) {
                            if (t.get(j - 1).equals("-")) {
                                temp += 0 - Integer.valueOf(t.get(j));
                            } else {
                                temp += Integer.valueOf(t.get(j));
                            }
                        }
                        temp += Integer.valueOf(t.get(0));
                    }
                    stack.push(String.valueOf(temp));
                }
            }
        }

        ArrayList<String> t = new ArrayList<String>();
        while (!stack.isEmpty()) {
            String elem = stack.pop();
            t.add(0, elem);
        }

        int temp = 0;
        for (int i = t.size() - 1; i > 0; i = i - 2) {
            if (t.get(i - 1).equals("-")) {
                temp += 0 - Integer.valueOf(t.get(i));
            } else {
                temp += Integer.valueOf(t.get(i));
            }
        }
        temp += Integer.valueOf(t.get(0));

        return temp;
    }


    /*
     *  Largest Number
     *  For example, given [3, 30, 34, 5, 9], the largest formed number is 9534330
     */
    public String largestNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for(int i=0; i<nums.length; i++) {
            strs[i] = String.valueOf(nums[i]);
        }

        Arrays.sort(strs, new Comparator<String>() {
            public int compare(String s1, String s2) {
                String leftRight = s1+s2;
                String rightLeft = s2+s1;
                return rightLeft.compareTo(leftRight);

            }
        });

        StringBuilder sb = new StringBuilder();
        for(String s: strs) {
            sb.append(s);
        }

        while(sb.charAt(0) == '0' && sb.length() > 1) {
            sb.deleteCharAt(0);
        }

        return sb.toString();
    }


    /*
     *  Different Ways to Add Parentheses
     *  Input: "2-1-1". ((2-1)-1) = 0 (2-(1-1)) = 2
     *  Output: [0, 2]
     */
    public List<Integer> diffWaysToCompute(String input) {
        List<Integer> result = new ArrayList<Integer>();
        if (input == null || input.length() == 0) {
            return result;
        }

        for (int i = 0; i < input.length(); i++) {
            char c = input.charAt(i);
            if (c != '+' && c != '-' && c != '*') {
                continue;
            }

            List<Integer> part1Result = diffWaysToCompute(input.substring(0, i));
            List<Integer> part2Result = diffWaysToCompute(input.substring(i + 1, input.length()));

            for (Integer m : part1Result) {
                for (Integer n : part2Result) {
                    if (c == '+') {
                        result.add(m + n);
                    } else if (c == '-') {
                        result.add(m - n);
                    } else if (c == '*') {
                        result.add(m * n);
                    }
                }
            }
        }

        if (result.size() == 0) {
            result.add(Integer.parseInt(input));
        }

        return result;
    }

    /*
     *  Search a 2D Matrix
     *  Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
     *
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int rNum = matrix.length;
        int cNum = matrix[0].length;
        for (int i=0; i<rNum; i++) {
            if (target >= matrix[i][0] && target <= matrix[i][cNum-1]) {
                for (int j=0; j<cNum; j++) {
                    if (matrix[i][j] == target) {
                        return true;
                    }
                }
            }
        }

        return false;
    }


    /*
     *  Search a 2D Matrix II
     *  Integers in each row are sorted in ascending from left to right.
     *  Integers in each column are sorted in ascending from top to bottom.
     */
    public boolean searchMatrixII(int[][] matrix, int target) {
        int m = matrix.length, n = matrix[0].length, i = 0, j = n-1;
        while (i<m && j>=0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] < target) {
                i++;
            } else {
                j--;
            }
        }
        return false;
    }

    /*
     *  Add Two Numbers
     *  Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
     *  Output: 7 -> 0 -> 8
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(-1);
        ListNode helper = dummyHead;
        int carry = 0;
        while (l1 != null || l2 != null) {
            if (l1 != null) {
                carry += l1.val;
                l1 = l1.next;
            }

            if (l2 != null) {
                carry += l2.val;
                l2 = l2.next;
            }

            helper.next = new ListNode(carry%10);
            carry /= 10;
            helper = helper.next;
        }

        if (carry == 1) {
            helper.next = new ListNode(1);
        }

        return dummyHead.next;
    }

    /*
     *  Longest Substring Without Repeating Characters
     *  "abcabcbb" is "abc"
     *  "bbbbb" the longest substring is "b",
     */
    public int lengthOfLongestSubstring(String s) {
        int[] charMap = new int[256];
        Arrays.fill(charMap, -1);
        int i = 0, maxLen = 0;
        for (int j=0; j<s.length(); j++) {
            if (charMap[s.charAt(j)] >= i) {
                i = charMap[s.charAt(j)] + 1;
            }
            charMap[s.charAt(j)] = j;
            maxLen = Math.max(j - i + 1, maxLen);
        }
        return maxLen;
    }

    /*
     *  Integer to English Words
     *  123 -> "One Hundred Twenty Three"
     *  12345 -> "Twelve Thousand Three Hundred Forty Five"
     */
    String[] lessThan20 = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
            "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    String[] tens = {"", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    String[] thousands = {"", "Thousand", "Million", "Billion"};

    public String numberToWords(int num) {
        if (num == 0) {
            return "Zero";
        }
        String result = "";
        int i = 0;
        while (num > 0) {
            if (num % 1000 != 0) {
                result = helper(num % 1000) + thousands[i] + " " + result;
            }
            num /= 1000;
            i++;
        }
        return result.trim();
    }

    private String helper(int num) {
        if (num == 0) {
            return "";
        } else if (num < 20) {
            return lessThan20[num] + " ";
        } else if (num < 100) {
            return tens[num / 10] + " " + helper(num % 10);
        } else {
            return lessThan20[num / 100] + " Hundred " + helper(num % 100);
        }
    }

    /*
     *  H-Index
     *  citations = [3, 0, 6, 1, 5]
     *  3 papers with at least 3 citations
     */
    public int hIndex(int[] citations) {
        int len = citations.length;
        int max = 0;
        for (int i=1; i<=len; i++) {
            int count = 0;
            for (int j=0; j<len; j++) {
                if (citations[j] >= i) {
                    count++;
                    if (count == i) {
                        break;
                    }
                }
            }
            if (count > max) {
                max = count;
            }
        }
        return max;
    }
    /*
     *  H-Index II
     *  citations = [3, 0, 6, 1, 5]
     *  3 papers with at least 3 citations
     */
    public int hIndex2(int[] citations) {
        if (citations == null || citations.length == 0) {
            return 0;
        }

        int lo = 0;
        int hi = citations.length - 1;
        int len = citations.length;

        while (lo + 1 < hi) {
            int mid = lo + (hi - lo) / 2;

            if (citations[mid] == len - mid) {
                return len - mid;
            } else if (citations[mid] < len - mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        if (citations[lo] >= len - lo) {
            return len - lo;
        }

        if (citations[hi] >= len - hi) {
            return len - hi;
        }

        return 0;
    }

    /*
     *  Spiral Matrix
     *  Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
     *
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        if(matrix == null || matrix.length == 0) {
            return result;
        }
        int m = matrix.length;
        int n = matrix[0].length;
        int x=0;
        int y=0;
        while (m > 0 && n > 0) {
            if (m == 1) {
                for(int i=0; i<n; i++) {
                    result.add(matrix[x][y++]);
                }
                break;
            } else if (n == 1) {
                for(int i=0; i<m; i++) {
                    result.add(matrix[x++][y]);
                }
                break;
            }
            for (int i=0; i<n-1; i++) {
                result.add(matrix[x][y++]);
            }

            for (int i=0; i<m-1; i++) {
                result.add(matrix[x++][y]);
            }

            for (int i=0; i<n-1; i++) {
                result.add(matrix[x][y--]);
            }

            for (int i=0; i<m-1; i++) {
                result.add(matrix[x--][y]);
            }
            x++;
            y++;
            m = m - 2;
            n = n - 2;
        }
        return result;
    }

    /*
     *  Rotate Image
     *  You are given an n x n 2D matrix representing an image.
     *  Rotate the image by 90 degrees (clockwise).
     */
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n / 2; i++) {
            for (int j = 0; j < Math.ceil(((double) n) / 2.); j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n-1-j][i];
                matrix[n-1-j][i] = matrix[n-1-i][n-1-j];
                matrix[n-1-i][n-1-j] = matrix[j][n-1-i];
                matrix[j][n-1-i] = temp;
            }
        }
    }

    /*
     *  Populating Next Right Pointers in Each Node
     *
     */
    public void connect(TreeLinkNode root) {
        if(root == null) {
            return;
        }

        if(root.left != null) {
            root.left.next = root.right;
        }
        if(root.right != null) {
            root.right.next = root.next==null ? null : root.next.left;
        }

        connect(root.left);
        connect(root.right);
    }


    /*
     *  Minimum Path Sum
     *  Given a m x n grid filled with non-negative numbers
     *   find a path from top left to bottom right which
     *   minimizes the sum of all numbers along its path
     */
    public int minPathSum(int[][] grid) {
        if(grid == null || grid.length == 0) {
            return 0;
        }

        int m = grid.length;
        int n = grid[0].length;

        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];

        // initialize top row
        for(int i=1; i<n; i++) {
            dp[0][i] = dp[0][i-1] + grid[0][i];
        }

        // initialize left column
        for(int j=1; j<m; j++) {
            dp[j][0] = dp[j-1][0] + grid[j][0];
        }

        // fill up the dp table
        for(int i=1; i<m; i++) {
            for(int j=1; j<n; j++) {
                if(dp[i-1][j] > dp[i][j-1]) {
                    dp[i][j] = dp[i][j-1] + grid[i][j];
                } else {
                    dp[i][j] = dp[i-1][j] + grid[i][j];
                }
            }
        }

        return dp[m-1][n-1];
    }

    /*
     *  Word Ladder
     *  beginWord = "hit"  endWord = "cog" wordList = ["hot","dot","dog","lot","log"]
     *  As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
     *  return its length 5.
     */
    public int ladderLength(String beginWord, String endWord, Set<String> wordList) {
        Queue<String> queue = new LinkedList<String>();
        int length = 1;
        queue.offer(beginWord);
        while(!queue.isEmpty()) {
            int size = queue.size();
            for (int i=0; i<size; i++) {
                String current = queue.poll();
                for (char c='a'; c<='z'; c++) {
                    for (int j=0; j<current.length(); j++) {
                        if (current.charAt(j) != c) {
                            String tmp = replace(current, j, c);
                            if (tmp.equals(endWord)) {
                                return length + 1;
                            } else if (wordList.contains(tmp)) {
                                queue.offer(tmp);
                                wordList.remove(tmp);
                            }
                        }
                    }
                }
            }
            length++;
        }
        return 0;
    }

    private String replace(String s, int index, char c) {
        char[] chars = s.toCharArray();
        chars[index] = c;
        return new String(chars);
    }


    /*
     *  Simplify Path
     *  path = "/home/", => "/home"
     *  path = "/a/./b/../../c/", => "/c"
     *
     */
    public String simplifyPath(String path) {
        if (path == null || path.length() == 0) {
            return path;
        }

        Stack<String> s = new Stack<String>();
        String[] arr = path.split("/");
        for (int i=0; i<arr.length; i++) {
            if (arr[i].equals(".") || arr[i].length() == 0) {
                continue;
            } else if (!arr[i].equals("..")) {
                s.push(arr[i]);
            } else if (arr[i].equals("..") && !s.isEmpty()) {
                s.pop();
            }
        }

        StringBuilder result = new StringBuilder();
        while (!s.isEmpty()) {
            result.insert(0, "/" + s.pop());
        }

        while (result.length() == 0) {
            result.insert(0, "/");
        }

        return result.toString();
    }

    /*
     *
     *   Odd Even Linked List
     *   Given 1->2->3->4->5->NULL,  return 1->3->5->2->4->NULL.
     */

    public ListNode oddEvenList(ListNode head) {
        ListNode preO = new ListNode(-1);
        ListNode preE = new ListNode(-1);
        ListNode runnerO = preO;
        ListNode runnerE = preE;
        while (head != null) {
            runnerO.next = new ListNode(head.val);
            runnerO = runnerO.next;
            if (head.next != null) {
                runnerE.next = new ListNode(head.next.val);
                runnerE = runnerE.next;
            }

            if (head.next != null) {
                head = head.next.next;
            } else {
                break;
            }


        }

        runnerE.next = null;
        runnerO.next = preE.next;

        return preO.next;

    }


    /*
     *  Product of Array Except Self
     *  For example, given [1,2,3,4], return [24,12,8,6].
     *
     */
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; i++) {
            res[i] = res[i - 1] * nums[i - 1];
        }
        int right = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] *= right;
            right *= nums[i];
        }
        return res;
    }


    /*
     *  Flatten Binary Tree to Linked List
     *  Given a binary tree, flatten it to a linked list in-place.
     *
     */

    public void flatten(TreeNode root) {
        if (root != null) {
            Stack<TreeNode> stack = new Stack<TreeNode>();
            stack.push(root);
            while (!stack.isEmpty()) {
                TreeNode temp = stack.pop();
                if (temp.right != null) {
                    stack.push(temp.right);
                }
                if (temp.left != null) {
                    stack.push(temp.left);
                }
                if (!stack.isEmpty()) {
                    temp.right = stack.peek();
                }
                temp.left = null;
            }
        }
    }

    /*
     *  Coin Change
     *  coins = [1, 2, 5], amount = 11
     *  return 3 (11 = 5 + 5 + 1)
     */

    public int coinChange(int[] coins, int amount) {
        if(amount < 1) return 0;
        int[] dp = new int[amount+1];
        int sum = 0;

        while(++sum <= amount) {
            int min = -1;
            for(int coin : coins) {
                if(sum >= coin && dp[sum-coin] != -1) {
                    int temp = dp[sum-coin]+1;
                    min = min < 0 ? temp : (temp < min ? temp : min);
                }
            }
            dp[sum] = min;
        }
        return dp[amount];
    }


    /*
     *  Basic Calculator II
     *  "3+2*2" = 7
     *
     */

    public int calculate2(String s) {
        int len;
        if(s==null || (len = s.length())==0) return 0;
        Stack<Integer> stack = new Stack<Integer>();
        int num = 0;
        char sign = '+';
        for(int i=0;i<len;i++){
            if(Character.isDigit(s.charAt(i))){
                num = num*10+s.charAt(i)-'0';
            }
            if((!Character.isDigit(s.charAt(i)) &&' '!=s.charAt(i)) || i==len-1){
                if(sign=='-'){
                    stack.push(-num);
                }
                if(sign=='+'){
                    stack.push(num);
                }
                if(sign=='*'){
                    stack.push(stack.pop()*num);
                }
                if(sign=='/'){
                    stack.push(stack.pop()/num);
                }
                sign = s.charAt(i);
                num = 0;
            }
        }

        int re = 0;
        for(int i:stack){
            re += i;
        }
        return re;
    }


    /*
     *  Restore IP Addresses
     *  Given "25525511135",
     *  return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
     */

    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<String>();
        int len = s.length();
        for(int i = 1; i<4 && i<len-2; i++) {
            for(int j = i+1; j<i+4 && j<len-1; j++) {
                for(int k = j+1; k<j+4 && k<len; k++) {
                    String s1 = s.substring(0,i), s2 = s.substring(i,j), s3 = s.substring(j,k), s4 = s.substring(k,len);
                    if(isValidIP(s1) && isValidIP(s2) && isValidIP(s3) && isValidIP(s4)) {
                        res.add(s1+"."+s2+"."+s3+"."+s4);
                    }
                }
            }
        }
        return res;
    }

    public boolean isValidIP(String s){
        if (s.length() > 3 || s.length() == 0 || (s.charAt(0) == '0' && s.length() > 1) || Integer.parseInt(s) > 255) {
            return false;
        }
        return true;
    }


    /*
     *  Maximum Product Subarray
     *  For example, given the array [2,3,-2,4],
     *  the contiguous subarray [2,3] has the largest product = 6.
     */

    public int maxProduct(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }

        int maxherepre = nums[0];
        int minherepre = nums[0];
        int maxsofar = nums[0];
        int maxhere, minhere;

        for (int i = 1; i < nums.length; i++) {
            maxhere = Math.max(Math.max(maxherepre * nums[i], minherepre * nums[i]), nums[i]);
            minhere = Math.min(Math.min(maxherepre * nums[i], minherepre * nums[i]), nums[i]);
            maxsofar = Math.max(maxhere, maxsofar);
            maxherepre = maxhere;
            minherepre = minhere;
        }
        return maxsofar;
    }


    /*
     *  Longest Increasing Subsequence
     *  Given [10, 9, 2, 5, 3, 7, 101, 18],
     *  The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4.
     */
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        int len = 0;

        for(int x : nums) {
            int i = Arrays.binarySearch(dp, 0, len, x);
            if(i < 0) i = -(i + 1);
            dp[i] = x;
            if(i == len) len++;
        }

        return len;
    }


    /*
     *  Peeking Iterator
     *  Given an Iterator class interface with methods: next() and hasNext()
     *  the element that will be returned by the next call to next().
     */
    class PeekingIterator implements Iterator<Integer> {

        private Integer next = null;
        private Iterator<Integer> iter;

        public PeekingIterator(Iterator<Integer> iterator) {
            // initialize any member here.
            iter = iterator;
            if (iter.hasNext()) {
                next = iterator.next();
            }
        }

        // Returns the next element in the iteration without advancing the iterator.
        public Integer peek() {
            return next;
        }


        // hasNext() and next() should behave the same as in the Iterator interface.
        // Override them if needed.
        @Override
        public Integer next() {
            int res = next;
            next = iter.hasNext()? iter.next() : null;
            return res;

        }

        @Override
        public boolean hasNext() {
            return next != null;
        }
    }


    /*
     *  Linked List Cycle II
     *  Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
     *
     */

    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head, start = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                while (slow != start) {
                    slow = slow.next;
                    start = start.next;
                }
                return start;
            }
        }
        return null;
    }


    /*
     *  Pow(x, n)
     *  Implement pow(x, n).
     *
     */
    public double myPow(double x, int n) {
        if(n == 0) {
            return 1;
        }

        if(n < 0) {
            n = -n;
            x = 1/x;
        }
        return (n%2 == 0) ? myPow(x*x, n/2) : x*myPow(x*x, n/2);
    }

     /*
     *  Group Anagrams
     *  For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
     *
     */
     public List<List<String>> groupAnagrams(String[] strs) {
         List<List<String>> result = new LinkedList<List<String>>();
         if (strs == null || strs.length == 0) {
             return result;
         }
         Arrays.sort(strs);
         Map<String, List<String>> map = new HashMap<String, List<String>>();
         for (String str : strs) {
             char[] arr = str.toCharArray();
             Arrays.sort(arr);
             String key = String.valueOf(arr);
             if (!map.containsKey(key)) {
                 List<String> list = new LinkedList<String>();
                 list.add(str);
                 map.put(key, list);
             } else {
                 List<String> temp = map.get(key);
                 temp.add(str);
                 map.put(key, temp);
             }
         }

         for (List<String> value : map.values()) {
             result.add(value);
         }

         return result;
     }


     /*
     *  First Missing Positive
     *  Given an unsorted integer array, find the first missing positive integer.
     *
     */

    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        for(int i = 0; i < n; i++) {
            while(nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            }
        }
        for(int i = 0; i < n; i++) {
            if(nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }


    /*
     *  Convert Sorted List to Binary Search Tree
     *  Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
     *
     */
    public TreeNode sortedListToBST(ListNode head) {
        return sortedListToBST(head, null);
    }


    private TreeNode sortedListToBST(ListNode head, ListNode tail) {
        if(head == null || head == tail) {
            return null;
        }
        if(head.next == tail) {
            return new TreeNode(head.val);
        }

        ListNode fast = head, slow = head;

        while (fast != tail && fast.next != tail) {
            fast = fast.next.next;
            slow = slow.next;
        }

        TreeNode root = new TreeNode(slow.val);
        root.left = sortedListToBST(head, slow);
        root.right = sortedListToBST(slow.next, tail);
        return root;
    }


    /*
     *  Convert Sorted Array to Binary Search Tree
     *  Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums.length == 0) {
            return null;
        }
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    public TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = (start + end) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBST(nums, start, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, end);
        return root;
    }


    /*
     *  Serialize and Deserialize Binary Tree
     *
     */
    public class Codec {

        private static final String spliter = ",";
        private static final String NN = "null";

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            buildString(root, sb);
            return sb.toString();
        }

        private void buildString(TreeNode node, StringBuilder sb) {
            if (node == null) {
                sb.append(NN).append(spliter);
            } else {
                sb.append(node.val).append(spliter);
                buildString(node.left, sb);
                buildString(node.right,sb);
            }
        }
        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            Queue<String> nodes = new LinkedList<>();
            nodes.addAll(Arrays.asList(data.split(spliter)));
            return buildTree(nodes);
        }

        private TreeNode buildTree(Queue<String> nodes) {
            String val = nodes.remove();
            if (val.equals(NN)) {
                return null;
            } else {
                TreeNode node = new TreeNode(Integer.valueOf(val));
                node.left = buildTree(nodes);
                node.right = buildTree(nodes);
                return node;
            }
        }
    }

    /*
     *  Reorder List
     *  Given a singly linked list L: L0→L1→…→Ln-1→Ln,
     *  reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…
     */
    public void reorderList(ListNode head) {
        if(head==null || head.next==null) {
            return;
        }

        //Find the middle of the list
        ListNode p1 = head;
        ListNode p2 = head;
        while(p2.next != null && p2.next.next != null) {
            p1 = p1.next;
            p2 = p2.next.next;
        }

        //Reverse the half after middle  1->2->3->4->5->6 to 1->2->3->6->5->4
        ListNode preMiddle = p1;
        ListNode preCurrent = p1.next;
        while (preCurrent.next != null) {
            ListNode current=preCurrent.next;
            preCurrent.next=current.next;
            current.next=preMiddle.next;
            preMiddle.next=current;
        }

        //Start reorder one by one  1->2->3->6->5->4 to 1->6->2->5->3->4
        p1 = head;
        p2 = preMiddle.next;
        while (p1 != preMiddle) {
            preMiddle.next=p2.next;
            p2.next=p1.next;
            p1.next=p2;
            p1=p2.next;
            p2=preMiddle.next;
        }
    }


    /*
     *  LRU Cache
     *  Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and set.
     *
     */
    public class LRUCache {
        private Map<Integer, Node> map;
        private Node head; // dummy "fence" head
        private Node tail; // dummy "fence" tail
        private int capacity;

        public LRUCache(int capacity) {
            this.capacity = capacity;
            map = new HashMap<Integer, Node>();
            head = new Node(-1, -1);
            tail = new Node(-1, -1);
            head.next = tail;
            tail.prev = head;
        }

        public int get(int key) {
            if( !map.containsKey(key) ) {
                return -1;
            }
            Node n = map.get(key);
            promoteToHead(n);
            return n.val;
        }

        public void set(int key, int value) {
            Node n;
            // update existing Node; does not alter cache size
            if( map.containsKey(key) ) {
                n = map.get(key);
                n.val = value;   // map.get(n.key) will now return node with new val
                promoteToHead(n);

                return;
            }
            if( map.size() == capacity ) {
                Node last = tail.prev;
                map.remove(last.key);
                remove(last);
            }
            n = new Node(key, value);
            addFirst(n);
            map.put(key, n);
        }

        /**
         * Move given Node to head of queue.
         */
        private void promoteToHead(Node n) {
            if( head != n ) {
                remove(n);
                addFirst(n);
            }
        }

        /**
         * Remove given Node from queue.
         */
        private void remove(Node n) {
            n.prev.next = n.next;
            n.next.prev = n.prev;
        }

        /**
         * Insert given Node to head of queue.
         */
        private void addFirst(Node n) {
            // first insert looks like:
            //  -1 <-> -1
            //  -1 <-> n <-> -1
            Node temp = head.next;
            head.next = n;
            n.prev = head;
            n.next = temp;
            n.next.prev = n;
        }

        public void printCache() throws Exception {
            if( head.next == tail ) {
                throw new Exception("empty cache!");
            }
            Node n = head.next;
            System.out.print("[ ");
            while( n != tail ) {
                System.out.print(n.val + " ");
                n = n.next;
            }
            System.out.println("]");
        }

        public class Node {
            int key;
            int val;
            Node prev;
            Node next;

            public Node(int key, int val) {
                this.key = key;
                this.val = val;
            }
        }
    }


    /**
     * Sqrt(x)
     */
    public int mySqrt(int x) {
        if (x == 0) {
            return 0;
        }
        int left = 1, right = Integer.MAX_VALUE;
        while (true) {
            int mid = left + (right - left)/2;
            if (mid > x/mid) {
                right = mid - 1;
            } else {
                if (mid + 1 > x/(mid + 1)) {
                    return mid;
                }
                left = mid + 1;
            }
        }
    }

    /**
     * Clone Graph
     */

    class UndirectedGraphNode {
        int label;
        List<UndirectedGraphNode> neighbors;
        UndirectedGraphNode(int x) { label = x; neighbors = new ArrayList<UndirectedGraphNode>(); }
    };

    public class CloneGraphSolution {
        private HashMap<Integer, UndirectedGraphNode> map = new HashMap<>();
        public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
            return clone(node);
        }
        private UndirectedGraphNode clone(UndirectedGraphNode node) {
            if (node == null) return null;

            if (map.containsKey(node.label)) {
                return map.get(node.label);
            }
            UndirectedGraphNode clone = new UndirectedGraphNode(node.label);
            map.put(clone.label, clone);
            for (UndirectedGraphNode neighbor : node.neighbors) {
                clone.neighbors.add(clone(neighbor));
            }
            return clone;
        }
    }


    /**
     * Count Primes
     */
    public int countPrimes(int n) {
        boolean[] notPrime = new boolean[n];
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (notPrime[i] == false) {
                count++;
                for (int j = 2; i*j < n; j++) {
                    notPrime[i*j] = true;
                }
            }
        }

        return count;
    }

    /**
     * Implement Stack using Queues
     */
    class MyStack {
        // Push element x onto stack.
        Queue<Integer> q = new LinkedList<Integer>();

        public void push(int x) {
            q.add(x);
        }

        // Removes the element on top of the stack.
        public void pop() {
            int size = q.size();
            for (int i=0; i<size-1; i++) {
                q.add(q.remove());
            }

            q.remove();
        }

        // Get the top element.
        public int top() {
            int size = q.size();
            for(int i = 0; i < size-1; i++) {
                q.add(q.remove());
            }
            int ret = q.remove();
            q.add(ret);
            return ret;
        }

        // Return whether the stack is empty.
        public boolean empty() {
            return q.isEmpty();
        }
    }


    /**
     * Kth Largest Element in an Array
     */
    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        int i = nums.length-1;
        while (k > 1 && i >=0) {
            i--;
            k--;
        }

        return nums[i];
    }


    /**
     *Given an array of integers and an integer k, find out whether
     * there are two distinct indices i and j in the array such that
     * nums[i] = nums[j] and the difference between i and j is
     *
     * 
     */
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Set<Integer> set = new HashSet<Integer>();
        for (int i=0; i<nums.length; i++) {
            if (i > k) {
                set.remove(nums[i-k-1]);
            }

            if (!set.contains(nums[i])) {
                set.add(nums[i]);
            } else {
                return true;
            }
        }

        return false;
    }







































}
