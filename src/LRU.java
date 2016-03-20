import java.util.HashMap;
import java.util.Map;

public class LRU {
    class Node {
        int key;
        int value;
        Node pre;
        Node next;
        public Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    private Map<Integer, Node> map;
    private int capacity;
    Node head, tail;

    public LRU(int capacity) {
        this.map = new HashMap<Integer, Node>();
        this.capacity = capacity;
        this.head = new Node(-1 ,-1);
        this.tail = new Node(-1 ,-1);
        head.next = tail;
        tail.pre = head;
    }

    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1;
        }
        Node n = map.get(key);
        promoteToHead(n);
        return n.value;
    }

    public void set(int key, int value) {
        if (map.containsKey(key)) {
            Node node = map.get(key);
            node.value = value;
            promoteToHead(node);
            return ;
        }

        if (map.size() == capacity) {
            Node last = tail.pre;
            map.remove(last.key);
            remove(last);
        }
        Node node = new Node(key, value);
        addFirst(node);
        map.put(key, node);
    }


    public void promoteToHead(Node n) {
        if (head != null) {
            remove(n);
            addFirst(n);
        }
    }

    public void remove(Node n) {
        n.pre.next = n.next;
        n.next.pre = n.pre;
    }

    public void addFirst(Node n) {
        Node temp = head.next;
        head.next = n;
        n.pre = head;
        n.next = temp;
        n.next.pre = n;
    }
}