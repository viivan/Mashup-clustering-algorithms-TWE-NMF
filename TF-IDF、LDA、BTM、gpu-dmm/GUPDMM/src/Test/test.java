package Test;

import java.io.*;
public class test {
    public static void main(String []args){
        String SystemPath = System.getProperty("user.dir");
        String path = "/resource/gd_data.txt";
        path = SystemPath + path;
        FileInputStream file = null;
        try {
            file = new FileInputStream(path);
            BufferedReader br = new BufferedReader(new InputStreamReader(file));
            String line = null;
            while((line = br.readLine()) != null){
                System.out.println(line);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
