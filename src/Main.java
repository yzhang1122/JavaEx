import javax.xml.soap.Node;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        Graph g = new Graph();
        g.addVertex('A');
        g.addVertex('B');
        g.addVertex('C');
        g.addVertex('D');
        g.addVertex('E');

        g.addEdge(0, 1); // AB
        g.addEdge(1, 2); // BC
        g.addEdge(0, 3); // AD
        g.addEdge(3, 4);
        System.out.print("Visits: ");
        g.dfs(); // depth-first search
        System.out.println();
        System.out.print("Visits: ");
        g.bfs(); // depth-first search
        System.out.println();
    }

    public static List<Integer> getRow(int rowIndex) {
        List<Integer> result = new LinkedList<Integer>();
        result.add(1);
        for (int i=1; i<=rowIndex; i++) {
            for (int j=i-1; j>=1; j--) {
                int temp = result.get(j-1) + result.get(j);
                result.set(j, temp);

            }
            result.add(1);
            System.out.println(result);
        }
        return result;
    }


    public static TreeNode deserialize(String data) {
        String[] arr = data.split(",");
        if (arr == null || arr.length == 0) {
            return null;
        }
        return null;
    }


    public static ListNode createList() {
        ListNode n1 = new ListNode(1);
        ListNode n2 = new ListNode(2);
        ListNode n3 = new ListNode(3);
        ListNode n4 = new ListNode(4);
        ListNode n5 = new ListNode(5);
        ListNode n6 = new ListNode(6);
        n1.next = n2;
        n2.next = n3;
        n3.next = n4;
        n4.next = n5;
        n5.next = n6;
        return n1;
    }

    public static void printList(ListNode head) {
        while (head.next != null) {
            System.out.print(head.val + ", ");
            head = head.next;
        }

        System.out.println(head.val);
    }

    public static void printArr(int[] arr) {
        int i = 0;
        while (i < arr.length-1) {
            System.out.print(arr[i] + ", ");
            i++;
        }
        System.out.println(arr[i]);
    }




}
