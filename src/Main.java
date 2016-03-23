import javax.xml.soap.Node;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        Car c1 = new Car();
        c1.setAge(4);
        c1.setName("aaaa");

        Car c2 = new Car();
        c2.setAge(4);
        c2.setName(null);

        List<Car> list = new LinkedList<>();
        list.add(c1);
        list.add(c2);

        Collections.sort(list);
        System.out.println(list);
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
