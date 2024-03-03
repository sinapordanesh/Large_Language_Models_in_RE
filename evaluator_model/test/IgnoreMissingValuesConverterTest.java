public class IgnoreMissingValuesConverterTest {

    @Test
    public void returns_null_if_value_is_missing()throws Exception {

        IgnoreMissingValuesConverter converter = new IgnoreMissingValuesConverter("M", "m");

        Assert.assertEquals(null, converter.convert(""));
        Assert.assertEquals(null, converter.convert("M"));
        Assert.assertEquals(null, converter.convert("m"));

        Assert.assertEquals(1.0f, converter.convert("1.0"), 1e-3);
    }
    
}