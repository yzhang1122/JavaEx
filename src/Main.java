import java.io.BufferedReader;
import java.io.FileReader;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.LinkedList;
import java.util.List;


public class Main {
  public static void main(String[] args) {
    DateTimeFormatter dateFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    DateTimeFormatter dateFormatter1 = DateTimeFormatter.ofPattern("MMddyyyy");
    LocalDate birth = LocalDate.parse("1960-12-30 00:00:00", dateFormatter);
    System.out.println(birth.format(dateFormatter1));

    printFileContent(
        "/Users/yzhang7/qbf-services/integration/src/test/resources/experian/ConsumerPremierNetConnectRequest.xml");
  }

  public static Integer addOne(Integer a) {
    return a + 1;
  }

  public static List<Integer> getRow(int rowIndex) {
    List<Integer> result = new LinkedList<Integer>();
    result.add(1);
    for (int i = 1; i <= rowIndex; i++) {
      for (int j = i - 1; j >= 1; j--) {
        int temp = result.get(j - 1) + result.get(j);
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
    while (i < arr.length - 1) {
      System.out.print(arr[i] + ", ");
      i++;
    }
    System.out.println(arr[i]);
  }

  public static void printFileContent(String fileName) {
    try {
      BufferedReader br = new BufferedReader(new FileReader(fileName));
      StringBuilder sb = new StringBuilder();
      String line = br.readLine();

      while (line != null) {
        sb.append(line);
        sb.append("\n");
        line = br.readLine();
      }

      String fileContent = sb.toString();
      System.out.println("---> file content: ");
      System.out.println(fileContent);

      String encodedFileContent =
          "&NETCONNECT_TRANSACTION=" + URLEncoder
              .encode(fileContent, StandardCharsets.UTF_8.toString());

      System.out.println("--->encoded file content: ");
      System.out.println(encodedFileContent);

    } catch (Exception e) {

    }
  }


}
