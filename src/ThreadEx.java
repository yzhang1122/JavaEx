

public class ThreadEx implements Runnable {
    private Thread t;
    private String threadName;

    public ThreadEx(String threadName) {
        this.threadName = threadName;
    }

    @Override
    public void run() {
        System.out.println("Running " +  threadName);
        try {
            for (int i=4; i>=0; i--) {
                System.out.println("Thread: " + threadName + ", " + i);
                Thread.sleep(1000);
            }
        } catch (Exception e) {
            System.out.println("Thread " +  threadName + " interrupted.");
        }
        System.out.println("Thread " +  threadName + " exiting.");
    }

    public void start() {
        System.out.println("Starting " +  threadName);
        if (t == null) {
            t = new Thread(this, threadName);
            t.start();
        }
    }
}


class ThreadExt extends Thread {
    private Thread t;
    private String threadName;

    public ThreadExt(String threadName) {
        this.threadName = threadName;
        System.out.println("Creating " +  threadName );
    }

    public void run() {
        System.out.println("Running " +  threadName);
        try {
            for (int i=5; i>=0; i--) {
                System.out.println("Thread: " + threadName + ", " + i);
                Thread.sleep(1000);
            }
        } catch (Exception e) {
            System.out.println("Thread " +  threadName + " interrupted.");
        }

        System.out.println("Thread " +  threadName + " exiting.");
    }

    public void start() {
        System.out.println("Starting " +  threadName);
        if (t == null) {
            t = new Thread (this, threadName);
            t.start ();
        }
    }
}


class ThreadExt1 extends Thread {
    private int startIdx, nThreads, maxIdx;

    public ThreadExt1(int s, int n, int m) {
        this.startIdx = s;
        this.nThreads = n;
        this.maxIdx = m;
    }
    @Override
    public void run() {
        for(int i = this.startIdx; i < this.maxIdx; i += this.nThreads) {
            try {
                Thread.sleep(1000);
                System.out.println("[ID " + this.getId() + "] " + i);
            } catch (Exception e) {
                //
            }
            System.out.println("done");
        }
    }
}


class ThreadExt2 implements Runnable {
    private int startIdx, nThreads, maxIdx;

    public ThreadExt2(int s, int n, int m) {
        this.startIdx = s;
        this.nThreads = n;
        this.maxIdx = m;
    }
    @Override
    public void run() {
        for(int i = this.startIdx; i < this.maxIdx; i += this.nThreads) {
            try {
                Thread.sleep(1000);
                System.out.println(i);
            } catch (Exception e) {
                //
            }
            System.out.println("done");
        }
    }
}

class Hello extends Thread {
    String name;
    public Hello(String name) {
        this.name = name;
    }
    public void run() {
        for (int i=0; i<=10; i++) {
            try {
                Thread.sleep(1000);
                System.out.println(name + ": " + i);
            } catch (Exception e) {

            }
        }
    }
}


class World extends Thread {
    String name;
    public World(String name) {
        this.name = name;
    }
    public void run() {
        for (int i=0; i<=10; i++) {
            try {
                Thread.sleep(1000);
                System.out.println(name + ": " + i);
            } catch (Exception e) {

            }

        }
    }
}