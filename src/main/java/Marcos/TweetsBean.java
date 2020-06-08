package Marcos;

import scala.Serializable;

public class TweetsBean implements Serializable {

    private static final long serialVersionUID = 1L;
    private	String	msg;
    private String	date;
    private String	source;
    private String	isRetweeted;
    private long	user_id;
    private long	followers;

    // Class Constructors
    public TweetsBean(){}

    public TweetsBean(String msg, String data, String source, String isRetweeted, long user_id, long followers){
        this.msg = msg;
        this.date = data;
        this.source = source;
        this.isRetweeted = isRetweeted;
        this.user_id = user_id;
        this.followers = followers;
    }

    public String getMsg() {return msg;}
    public void setMsg(String msg) {this.msg = msg;}

    public String getDate() {return date;}
    public void setDate(String date) {this.date = date;}

    public String getSource() {return source;}
    public void setSource(String source) {this.source = source;}

    public String getIsRetweeted() {return isRetweeted;}
    public void setIsRetweeted(String isRetweeted) {this.isRetweeted = isRetweeted;}

    public long getUser_id() {return user_id;}
    public void setUser_id(long user_id) {this.user_id = user_id;}

    public long getFollowers() {return followers;}
    public void setFollowers(long followers) {this.followers = followers;}

}
