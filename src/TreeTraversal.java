

import java.util.ArrayDeque;
import java.util.LinkedList;
import java.util.List;

public class TreeTraversal {
    public static void main (String[] args) {

        TreeNode root = makeBinaryTreeByArray(new int[] {0,1,2,3,4,5,6}, 1);
        System.out.println(AllPath(root));
    }

    public static TreeNode makeBinaryTreeByArray(int[] arr,int index) {
        if (index < arr.length) {
            if (arr[index] != 0) {
                TreeNode node = new TreeNode(arr[index]);
                arr[index] = 0;
                node.left = makeBinaryTreeByArray(arr, index*2);
                node.right = makeBinaryTreeByArray(arr, index*2+1);
                return node;
            }
        }

        return null;
    }



    public static void DFS(TreeNode root) {

        if (root == null) {
            return;
        }

        ArrayDeque<TreeNode> stack = new ArrayDeque<TreeNode>();
        stack.push(root);

        while (!stack.isEmpty()) {
            TreeNode node = stack.pop();
            System.out.print(node.val+ "  ");
            if (node.right != null) {
                stack.push(node.right);
            }

            if (node.left != null) {
                stack.push(node.left);
            }
        }
    }

    public static void BFS(TreeNode root) {
        if (root == null) {
            return;
        }

        ArrayDeque<TreeNode> stack = new ArrayDeque<TreeNode>();
        stack.push(root);

        while (!stack.isEmpty()) {
            TreeNode node = stack.remove();
            System.out.print(node.val+ "  ");
            if (node.left != null) {
                stack.add(node.left);
            }

            if (node.right != null) {
                stack.add(node.right);
            }
        }
    }


    public static List<List<Integer>> AllPath(TreeNode root) {
        List<List<Integer>> result = new LinkedList<List<Integer>>();
        if (root == null) {
            return result;
        }

        List<Integer> list = new LinkedList<Integer>();
        list.add(root.val);
        DFS1(root, result, list);
        return result;

    }


    public static void DFS1(TreeNode t, List<List<Integer>> result, List<Integer> list) {
        if (t.left == null && t.right == null) {
            List<Integer> temp = new LinkedList<Integer>();
            temp.addAll(list);
            result.add(temp);
            return;
        }

        if (t.left != null) {
            list.add(t.left.val);
            DFS1(t.left, result, list);
            list.remove(list.size() - 1);
        }

        if (t.right != null) {
            list.add(t.right.val);
            DFS1(t.right, result, list);
            list.remove(list.size() - 1);
        }
    }

    public static void createBST(TreeNode root, int val) {
        TreeNode node = root;
        while (true) {
            if (val < node.val) {
                if (node.left == null) {
                    node.left = new TreeNode(val);
                    break;
                }
                node = node.left;
            } else {
                if (node.right == null) {
                    node.right = new TreeNode(val);
                    break;
                }
                node = node.right;

            }
        }
    }

    public static void inorderTraverse(TreeNode node, List<Integer> list) {
        if (node.left != null) {
            inorderTraverse(node.left, list);
        }
        list.add(node.val);
        if (node.right != null) {
            inorderTraverse(node.right, list);
        }
    }



















}
