import java.util.LinkedList;
import java.util.Queue;
import java.util.Stack;

class Vertex {
    public char label;
    public boolean isVistied;
    public Vertex(char label) {
        this.label = label;
        isVistied = false;
    }
}

public class Graph {
    private final int MAX_VERTS = 20;
    private Vertex vertexList[];
    private int adjMat[][];
    private int nVerts;
    public Graph() {
        vertexList = new Vertex[MAX_VERTS];
        adjMat = new int[MAX_VERTS][MAX_VERTS];
        nVerts = 0;
        for (int i=0; i<MAX_VERTS; i++) {
            for (int j=0; j<MAX_VERTS; j++) {
                adjMat[i][i] = 0;
            }
        }
    }

    public void addVertex(char lab) {
        vertexList[nVerts++] = new Vertex(lab);
    }

    public void addEdge(int start, int end) {
        adjMat[start][end] = 1;
        adjMat[end][start] = 1;
    }

    public void displayVertex(int v) {
        System.out.print(vertexList[v].label);
    }

    public int getAdjUnvisitedVertex(int v) {
        for (int i=0; i<nVerts; i++) {
            if (adjMat[v][i] == 1 && vertexList[i].isVistied == false) {
                return i;
            }
        }
        return -1;
    }

    public void dfs() {
        Stack<Integer> s = new Stack<>();
        vertexList[0].isVistied = true;
        displayVertex(0);
        s.push(0);
        while (!s.isEmpty()) {
            int v = getAdjUnvisitedVertex(s.peek());
            if (v == -1) {
                s.pop();
            } else {
                vertexList[v].isVistied = true;
                displayVertex(v);
                s.push(v);
            }
        }

        for (int i=0; i<nVerts; i++) {
            vertexList[i].isVistied = false;
        }
    }

    public void bfs() {
        Queue<Integer> q = new LinkedList<>();
        vertexList[0].isVistied = true;
        displayVertex(0);
        q.add(0);
        while (!q.isEmpty()) {
            int v1 = q.remove();
            while (getAdjUnvisitedVertex(v1) != -1) {
                int v2 = getAdjUnvisitedVertex(v1);
                vertexList[v2].isVistied = true;
                displayVertex(v2);
                q.add(v2);
            }
        }

        for (int i=0; i<nVerts; i++) {
            vertexList[i].isVistied = false;
        }
    }















}
