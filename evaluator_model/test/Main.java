public class Main {
    public Float convert(final String s) {

        if(StringUtils.isNullOrWhiteSpace(s)) {
            return null;
        }

        boolean isMissingValue = missingValueRepresentation
                .stream()
                .anyMatch(x -> x.equals(s));

        if(isMissingValue) {
            return null;
        }

        return Float.parseFloat(s);
    }
}