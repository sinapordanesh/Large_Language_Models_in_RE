/*
 * Copyright 2015 Ben Manes. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.benmanes.caffeine.cache.simulator;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.collect.Sets.toImmutableEnumSet;
import static java.util.Locale.US;
import static java.util.Objects.requireNonNull;

import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Pattern;

import org.checkerframework.checker.nullness.qual.Nullable;

import com.github.benmanes.caffeine.cache.simulator.admission.Admission;
import com.github.benmanes.caffeine.cache.simulator.membership.FilterType;
import com.github.benmanes.caffeine.cache.simulator.parser.TraceFormat;
import com.github.benmanes.caffeine.cache.simulator.report.ReportFormat;
import com.google.common.base.CaseFormat;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigException;

/**
 * The simulator's configuration. A policy can extend this class as a convenient way to extract
 * its own settings.
 *
 * @author ben.manes@gmail.com (Ben Manes)
 */
public class BasicSettings {
  private static final Pattern NUMERIC_SEPARATOR = Pattern.compile("[_,]");

  private final Config config;

  public BasicSettings(Config config) {
    this.config = requireNonNull(config);
  }

  public ActorSettings actor() {
    return new ActorSettings();
  }

  public ReportSettings report() {
    return new ReportSettings();
  }

  public int randomSeed() {
    return getFormattedInt("random-seed");
  }

  public Set<String> policies() {
    return config().getStringList("policies").stream()
        .map(policy -> policy.toLowerCase(US))
        .collect(toImmutableSet());
  }

  public Set<Admission> admission() {
    return config().getStringList("admission").stream()
        .map(policy -> policy.toUpperCase(US))
        .map(Admission::valueOf)
        .collect(toImmutableEnumSet());
  }

  public MembershipSettings membership() {
    return new MembershipSettings();
  }

  public TinyLfuSettings tinyLfu() {
    return new TinyLfuSettings();
  }

  public long maximumSize() {
    return getFormattedLong("maximum-size");
  }

  public TraceSettings trace() {
    return new TraceSettings();
  }

  /** Returns the config resolved at the simulator's path. */
  public Config config() {
    return config;
  }

  /** Gets the quoted integer at the given path, ignoring comma and underscore separators */
  protected int getFormattedInt(String path) {
    return parseFormattedNumber(path, config()::getInt, Ints::tryParse);
  }

  /** Gets the quoted long at the given path, ignoring comma and underscore separators */
  protected long getFormattedLong(String path) {
    return parseFormattedNumber(path, config()::getLong, Longs::tryParse);
  }

  private <T extends Number> T parseFormattedNumber(String path,
      Function<String, T> getter, Function<String, @Nullable T> tryParse) {
    try {
      return getter.apply(path);
    } catch (ConfigException.Parse | ConfigException.WrongType e) {
      var matcher = NUMERIC_SEPARATOR.matcher(config().getString(path));
      var value = tryParse.apply(matcher.replaceAll(""));
      if (value == null) {
        throw e;
      }
      return value;
    }
  }

  public final class ActorSettings {
    public int mailboxSize() {
      return config().getInt("actor.mailbox-size");
    }
    public int batchSize() {
      return config().getInt("actor.batch-size");
    }
  }

  public final class ReportSettings {
    public ReportFormat format() {
      return ReportFormat.valueOf(config().getString("report.format").toUpperCase(US));
    }
    public String sortBy() {
      return config().getString("report.sort-by").trim();
    }
    public boolean ascending() {
      return config().getBoolean("report.ascending");
    }
    public String output() {
      return config().getString("report.output").trim();
    }
  }

  public final class MembershipSettings {
    public FilterType filter() {
      String type = config().getString("membership.filter");
      return FilterType.valueOf(CaseFormat.LOWER_HYPHEN.to(CaseFormat.UPPER_UNDERSCORE, type));
    }
    public long expectedInsertions() {
      double multiplier = config().getDouble("membership.expected-insertions-multiplier");
      return (long) (maximumSize() * multiplier);
    }
    public double fpp() {
      return config().getDouble("membership.fpp");
    }
  }

  public final class TinyLfuSettings {
    public String sketch() {
      return config().getString("tiny-lfu.sketch");
    }
    public boolean conservative() {
      return config().getBoolean("tiny-lfu.count-min.conservative");
    }
    public JitterSettings jitter() {
      return new JitterSettings();
    }
    public CountMin4Settings countMin4() {
      return new CountMin4Settings();
    }
    public CountMin64Settings countMin64() {
      return new CountMin64Settings();
    }

    public final class JitterSettings {
      public boolean enabled() {
        return config().getBoolean("tiny-lfu.jitter.enabled");
      }
      public int threshold() {
        return config().getInt("tiny-lfu.jitter.threshold");
      }
      public double probability() {
        return config().getDouble("tiny-lfu.jitter.probability");
      }
    }
    public final class CountMin4Settings {
      public String reset() {
        return config().getString("tiny-lfu.count-min-4.reset");
      }
      public double countersMultiplier() {
        return config().getDouble("tiny-lfu.count-min-4.counters-multiplier");
      }
      public IncrementalSettings incremental() {
        return new IncrementalSettings();
      }
      public PeriodicSettings periodic() {
        return new PeriodicSettings();
      }
      public final class IncrementalSettings {
        public int interval() {
          return config().getInt("tiny-lfu.count-min-4.incremental.interval");
        }
      }
      public final class PeriodicSettings {
        public DoorkeeperSettings doorkeeper() {
          return new DoorkeeperSettings();
        }
      }
    }
    public final class CountMin64Settings {
      public double eps() {
        return config().getDouble("tiny-lfu.count-min-64.eps");
      }
      public double confidence() {
        return config().getDouble("tiny-lfu.count-min-64.confidence");
      }
    }
    public final class DoorkeeperSettings {
      public boolean enabled() {
        return config().getBoolean("tiny-lfu.count-min-4.periodic.doorkeeper.enabled");
      }
    }
  }

  public final class TraceSettings {
    public long skip() {
      return getFormattedLong("trace.skip");
    }
    public long limit() {
      return config().getIsNull("trace.limit") ? Long.MAX_VALUE : getFormattedLong("trace.limit");
    }
    public boolean isFiles() {
      return config().getString("trace.source").equals("files");
    }
    public boolean isSynthetic() {
      return config().getString("trace.source").equals("synthetic");
    }
    public TraceFilesSettings traceFiles() {
      checkState(isFiles());
      return new TraceFilesSettings();
    }
    public SyntheticSettings synthetic() {
      checkState(isSynthetic());
      return new SyntheticSettings();
    }
  }

  public final class TraceFilesSettings {
    public List<String> paths() {
      return config().getStringList("files.paths");
    }
    public TraceFormat format() {
      return TraceFormat.named(config().getString("files.format"));
    }
  }

  public final class SyntheticSettings {
    public String distribution() {
      return config().getString("synthetic.distribution");
    }
    public int events() {
      return config().getInt("synthetic.events");
    }
    public CounterSettings counter() {
      return new CounterSettings();
    }
    public RepeatSettings repeating() {
      return new RepeatSettings();
    }
    public UniformSettings uniform() {
      return new UniformSettings();
    }
    public ExponentialSettings exponential() {
      return new ExponentialSettings();
    }
    public HotspotSettings hotspot() {
      return new HotspotSettings();
    }
    public ZipfianSettings zipfian() {
      return new ZipfianSettings();
    }

    public final class CounterSettings {
      public int start() {
        return config().getInt("synthetic.counter.start");
      }
    }
    public final class RepeatSettings {
      public int items() {
        return config().getInt("synthetic.repeating.items");
      }
    }
    public final class UniformSettings {
      public int lowerBound() {
        return config().getInt("synthetic.uniform.lower-bound");
      }
      public int upperBound() {
        return config().getInt("synthetic.uniform.upper-bound");
      }
    }
    public final class ExponentialSettings {
      public double mean() {
        return config().getDouble("synthetic.exponential.mean");
      }
    }
    public final class HotspotSettings {
      public int lowerBound() {
        return config().getInt("synthetic.hotspot.lower-bound");
      }
      public int upperBound() {
        return config().getInt("synthetic.hotspot.upper-bound");
      }
      public double hotsetFraction() {
        return config().getDouble("synthetic.hotspot.hotset-fraction");
      }
      public double hotOpnFraction() {
        return config().getDouble("synthetic.hotspot.hot-opn-fraction");
      }
    }
    public final class ZipfianSettings {
      public int items() {
        return config().getInt("synthetic.zipfian.items");
      }
      public double constant() {
        return config().getDouble("synthetic.zipfian.constant");
      }
    }
  }
}
