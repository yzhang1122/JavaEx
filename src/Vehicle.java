import java.util.Comparator;

public abstract class Vehicle {
    final int avc = 21;
    static int b = 2;
    public abstract void show();

    public Vehicle() {

    }

    protected void show1() {

    }
    private void sss() {

    }

}

class Car implements Comparable<Car> {
    String name;
    int age;

    public void setName(String name) {
        this.name = name;
    }

    public void setAge(int age) {
        this.age = age;
    }

    @Override
    public int compareTo(Car o) {
        if (this.age != o.age) {
            return this.age > o.age? 1 : -1;
        } else {
            return this.name.compareTo(o.name);
        }
    }

    @Override
    public String toString() {
        return "Car{" +
                "name='" + name + '\'' +
                ", age=" + age +
                '}';
    }
}

