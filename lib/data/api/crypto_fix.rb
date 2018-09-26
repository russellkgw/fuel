class Data::Api::CryptoFix
  def self.insert_averages

    btc_fixes = compute_btc_fixes
    ltc_fixes = compute_ltc_fixes

    # byebug

    btc_fixes.each do |date, vals|
      next if BtcPrice.where(date: date).any?

      BtcPrice.create({ date: date,
                        mid: vals[:mid],
                        volume: vals[:volume] })
    end

    ltc_fixes.each do |date, vals|
      next if LtcPrice.where(date: date).any?

      LtcPrice.create({ date: date,
                        mid: vals[:mid],
                        volume: vals[:volume] })  # add high low last bid ask
    end
  end

  def self.compute_btc_fixes
    missing_date = []
    btc_prices = BtcPrice.where("date >= '2014-04-15'").order(date: :asc).all

    (Date.new(2014, 4, 15)..(btc_prices.last.date)).each do |date|
      missing_date << date if btc_prices.select { |x| x.date == date }.empty?
    end

    fixes = {}

    missing_date.each do |date|
      before = BtcPrice.where("date < ?", date).order(date: :desc).first
      after = BtcPrice.where("date > ?", date).order(date: :asc).first

      if before.present? && after.present?
        fixes[date.to_s] = {mid: (before.mid + after.mid) / 2.0, volume: (before.volume + after.volume) / 2.0}
      end
    end

    fixes
  end

  def self.compute_ltc_fixes
    missing_date = []
    ltc_prices = LtcPrice.where("date >= '2014-04-15'").order(date: :asc).all

    (Date.new(2014, 4, 15)..(ltc_prices.last.date)).each do |date|
      missing_date << date if ltc_prices.select { |x| x.date == date }.empty?
    end

    fixes = {}

    missing_date.each do |date|
      before = LtcPrice.where("date < ?", date).order(date: :desc).first
      after = LtcPrice.where("date > ?", date).order(date: :asc).first

      if before.present? && after.present?
        fixes[date.to_s] = {mid: (before.mid + after.mid) / 2.0, volume: (before.volume + after.volume) / 2.0}
      end
    end

    fixes
  end
end