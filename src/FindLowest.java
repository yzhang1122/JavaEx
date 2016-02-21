package algorithms;

class Address {
    public int value;
    public Address[] children;
    public Address(int value) {
        this.value = value;
    }

}

public class FindLowest {

    public Address root = new Address(12);
    public Address input1 = new Address(-144);
    public Address input2 = new Address(2);
    public Address input3 = new Address(1);
    public Address input4 = new Address(-1);
    public Address input5 = new Address(-150);

    public void setInput() {
        input3.children = new Address[] {input4, input5};
        root.children = new Address[] {input1, input2, input3};

    }

    public void findLowest(Address root, Address min) {

        if (min.value > root.value) {
            min.value = root.value;
        }

        if (root.children != null) {
            for (Address child : root.children) {
                findLowest(child, min);
            }
        }
    }

}
